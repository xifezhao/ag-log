import os
import mmap
import struct
import time
import random
import string
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from abc import ABC, abstractmethod

# ==============================================================================
# 1. CONFIGURATION
# ==============================================================================
class Config:
    PAGE_SIZE = 4096  # 4 KB page size
    GRANULARITY_THRESHOLD = 128  # For AG-Log, in bytes
    
    NUM_RECORDS_IN_DB = 10000
    NUM_OPERATIONS_PER_RUN = 10000

    # NVM Emulation paths
    NVM_FILE_PATH = "nvm.dat"
    LOG_FILE_PATH = "wal.log"

    # YCSB Workload parameters
    YCSB_UPDATE_VALUE_SIZE_SMALL = 64
    YCSB_UPDATE_VALUE_SIZE_LARGE = 1024

# ==============================================================================
# 2. EMULATED NVM
# ==============================================================================
class EmulatedNVM:
    """Simulates NVM using a memory-mapped file and tracks writes."""
    def __init__(self, file_path, size_in_bytes):
        self.file_path = file_path
        self.size = size_in_bytes
        self.bytes_written = 0
        
        dir_name = os.path.dirname(file_path)
        if dir_name and not os.path.exists(dir_name):
            os.makedirs(dir_name)
        
        if os.path.exists(self.file_path):
            os.remove(self.file_path)
            
        self.fd = os.open(file_path, os.O_CREAT | os.O_RDWR)
        os.ftruncate(self.fd, self.size)
        self.mmap = mmap.mmap(self.fd, self.size)

    def write(self, offset, data):
        data_len = len(data)
        if offset + data_len > self.size:
            raise ValueError(f"Write exceeds NVM size: offset={offset}, len={data_len}, size={self.size}")
        self.mmap[offset:offset + data_len] = data
        self.bytes_written += data_len
        
    def read(self, offset, length):
        return self.mmap[offset:offset+length]

    def flush(self, offset, length):
        """
        Flushes the memory map to disk. For cross-platform compatibility, we flush the whole file.
        """
        self.mmap.flush()

    def get_bytes_written(self):
        return self.bytes_written

    def close(self):
        self.mmap.close()
        os.close(self.fd)
        if os.path.exists(self.file_path):
            os.remove(self.file_path)

# ==============================================================================
# 3. WRITE-AHEAD LOGGING IMPLEMENTATIONS
# ==============================================================================
class AbstractWAL(ABC):
    """Abstract Base Class for Write-Ahead Logging."""
    def __init__(self, log_nvm):
        self.log_nvm = log_nvm
        self.log_tail = 0

    def _append_log(self, record):
        offset = self.log_tail
        record_len = len(record)
        self.log_nvm.write(offset, record)
        self.log_nvm.flush(offset, record_len)
        self.log_tail += record_len

    @abstractmethod
    def log_update(self, page_addr, offset_in_page, old_data, new_data, full_page_data):
        pass

    def commit(self, txn_id):
        record = struct.pack("!B Q", 1, txn_id)  # Type 1 = COMMIT
        self._append_log(record)
    
    def recover(self):
        current_pos = 0
        while current_pos < self.log_tail:
            try:
                header_data = self.log_nvm.read(current_pos, 1) # Read type
                if not header_data: break
                
                record_type, = struct.unpack("!B", header_data)
                record_len = 0
                
                if record_type == 1: # COMMIT
                    record_len = 9
                elif record_type == 2: # PAGE
                    record_len = 9 + Config.PAGE_SIZE
                elif record_type == 3: # DELTA
                    len_header = self.log_nvm.read(current_pos + 11, 2)
                    data_len, = struct.unpack("!H", len_header)
                    record_len = 13 + 2 * data_len
                else:
                     break
                
                self.log_nvm.read(current_pos, record_len)
                current_pos += record_len
            except (struct.error, IndexError):
                break
        return

class PageWAL(AbstractWAL):
    """Logs the entire page for every update."""
    def log_update(self, page_addr, offset_in_page, old_data, new_data, full_page_data):
        record_type = 2
        header = struct.pack("!B Q", record_type, page_addr)
        self._append_log(header + full_page_data)

class DeltaWAL(AbstractWAL):
    """Logs only the delta (changes) for every update."""
    def log_update(self, page_addr, offset_in_page, old_data, new_data, full_page_data):
        record_type = 3
        data_len = len(new_data)
        header = struct.pack("!B Q H H", record_type, page_addr, offset_in_page, data_len)
        self._append_log(header + old_data + new_data)

class AGLog(AbstractWAL):
    """Adaptively chooses between Page and Delta logging."""
    def log_update(self, page_addr, offset_in_page, old_data, new_data, full_page_data):
        if len(new_data) < Config.GRANULARITY_THRESHOLD:
            record_type = 3
            data_len = len(new_data)
            header = struct.pack("!B Q H H", record_type, page_addr, offset_in_page, data_len)
            self._append_log(header + old_data + new_data)
        else:
            record_type = 2
            header = struct.pack("!B Q", record_type, page_addr)
            self._append_log(header + full_page_data)

# ==============================================================================
# 4. KEY-VALUE STORE
# ==============================================================================
class PagedKVStore:
    """A simplified paged KV store that uses a WAL."""
    def __init__(self, data_nvm, wal_impl):
        self.data_nvm = data_nvm
        self.wal = wal_impl
        self.volatile_cache = {}

    def put(self, key, value):
        page_addr, offset_in_page = self._get_addr(key)
        
        old_data = self.get(key)
        old_data_encoded = old_data.encode().ljust(len(value.encode()))

        full_page_data = self.data_nvm.read(page_addr, Config.PAGE_SIZE)

        self.wal.log_update(page_addr, offset_in_page, old_data_encoded, value.encode(), full_page_data)
        self.volatile_cache[key] = value

    def get(self, key):
        return self.volatile_cache.get(key, "")

    def commit(self, txn_id):
        self.wal.commit(txn_id)

    def _get_addr(self, key):
        h = hash(key)
        page_index = h % (self.data_nvm.size // Config.PAGE_SIZE)
        page_addr = page_index * Config.PAGE_SIZE
        offset_in_page = (h >> 16) % (Config.PAGE_SIZE - 512)
        return page_addr, offset_in_page

# ==============================================================================
# 5. WORKLOAD GENERATORS
# ==============================================================================
def generate_random_string(length):
    return ''.join(random.choices(string.ascii_letters + string.digits, k=length))

def ycsb_workload(name, num_ops, keys, read_prop, update_prop, value_size):
    ops = []
    for _ in range(num_ops):
        op_type = random.choices(["get", "put"], [read_prop, update_prop])[0]
        key = random.choice(keys)
        if op_type == "get":
            ops.append(("get", key))
        else:
            value = generate_random_string(value_size)
            ops.append(("put", key, value))
    return {"name": name, "ops": ops}

def tpcc_simplified_workload(num_txns, keys):
    ops = []
    for i in range(num_txns):
        ops.append(("txn_begin", i))
        ops.append(("put", random.choice(keys), generate_random_string(100)))
        for _ in range(random.randint(5, 10)):
            ops.append(("put", random.choice(keys), generate_random_string(24)))
        ops.append(("txn_commit", i))
    return {"name": "TPC-C (Simplified)", "ops": ops}

# ==============================================================================
# 6. EXPERIMENT RUNNER
# ==============================================================================
def run_experiment(wal_class, workload):
    print(f"--- Running {workload['name']} on {wal_class.__name__} ---")
    
    log_nvm = EmulatedNVM(Config.LOG_FILE_PATH, size_in_bytes=1024*1024*200) # 200MB log
    data_nvm = EmulatedNVM(Config.NVM_FILE_PATH, size_in_bytes=1024*1024*50) # 50MB data
    
    wal = wal_class(log_nvm)
    store = PagedKVStore(data_nvm, wal)
    
    logical_bytes_written = 0
    op_count = 0

    start_time = time.perf_counter()
    for op in workload['ops']:
        if op[0] == "put":
            store.put(op[1], op[2])
            logical_bytes_written += len(op[2])
            op_count += 1
        elif op[0] == "get":
            store.get(op[1])
            op_count += 1
        elif op[0] == "txn_commit":
            store.commit(op[1])
    end_time = time.perf_counter()
    
    duration = end_time - start_time
    throughput = op_count / duration if duration > 0 else float('inf')
    latency = (duration / op_count) * 1000 if op_count > 0 else 0
    
    physical_bytes_written = log_nvm.get_bytes_written()
    waf = physical_bytes_written / logical_bytes_written if logical_bytes_written > 0 else 0
    
    recovery_start_time = time.perf_counter()
    wal.recover()
    recovery_time = time.perf_counter() - recovery_start_time
    
    log_nvm.close()
    data_nvm.close()
    
    return {
        "WAL Type": wal_class.__name__,
        "Workload": workload['name'],
        "Throughput (ops/s)": throughput,
        "Latency (ms/op)": latency,
        "Write Amplification (WAF)": waf,
        "Recovery Time (s)": recovery_time
    }

def plot_results(df):
    """Generates and saves plots as PDF based on the results DataFrame."""
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
        
    sns.set_theme(style="whitegrid")
    
    metrics = ["Throughput (ops/s)", "Write Amplification (WAF)", "Latency (ms/op)", "Recovery Time (s)"]
    # *** CHANGE HERE: Use .pdf extension ***
    filenames = ["fig5_throughput.pdf", "fig7_waf.pdf", "fig6_latency.pdf", "fig9_recovery.pdf"]

    for metric, filename in zip(metrics, filenames):
        plt.figure(figsize=(12, 7))
        ax = sns.barplot(data=df, x="Workload", y=metric, hue="WAL Type")
        ax.set_title(f"{metric} by Workload and WAL Type", fontsize=16)
        ax.set_xticklabels(ax.get_xticklabels(), rotation=15, ha="right")
        plt.tight_layout()
        plt.savefig(os.path.join(results_dir, filename))
        print(f"Saved plot: {os.path.join(results_dir, filename)}")
        plt.close()

if __name__ == "__main__":
    keys = [f"user{i}" for i in range(Config.NUM_RECORDS_IN_DB)]
    
    workloads_to_run = [
        ycsb_workload("YCSB-A (small)", Config.NUM_OPERATIONS_PER_RUN, keys, 0.5, 0.5, Config.YCSB_UPDATE_VALUE_SIZE_SMALL),
        ycsb_workload("YCSB-B (small)", Config.NUM_OPERATIONS_PER_RUN, keys, 0.95, 0.05, Config.YCSB_UPDATE_VALUE_SIZE_SMALL),
        ycsb_workload("YCSB-A (large)", Config.NUM_OPERATIONS_PER_RUN, keys, 0.5, 0.5, Config.YCSB_UPDATE_VALUE_SIZE_LARGE),
        tpcc_simplified_workload(Config.NUM_OPERATIONS_PER_RUN // 8, keys),
    ]
    
    wal_implementations = [PageWAL, DeltaWAL, AGLog]
    
    all_results = []
    for wal_class in wal_implementations:
        for workload in workloads_to_run:
            result = run_experiment(wal_class, workload)
            all_results.append(result)
            print(f"Result: {result}")

    results_df = pd.DataFrame(all_results)
    
    results_dir = "results"
    if not os.path.exists(results_dir):
        os.makedirs(results_dir)
    results_df.to_csv(os.path.join(results_dir, "experiment_results.csv"), index=False)
    print(f"\nResults saved to {os.path.join(results_dir, 'experiment_results.csv')}")
    print(results_df)

    plot_results(results_df)