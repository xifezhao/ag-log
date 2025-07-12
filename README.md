# AG-Log: An Adaptive Granularity Write-Ahead Logging System for Persistent Memory

This repository contains the Python implementation of the experiments described in the paper, *"AG-Log: An Adaptive Granularity Write-Ahead Logging System for Persistent Memory."* The project aims to reproduce the core ideas of the paper, which addresses the performance bottlenecks of traditional Write-Ahead Logging (WAL) on Persistent Memory (PM) through an adaptive logging strategy.

## Abstract

The emergence of Persistent Memory (PM) presents a transformative opportunity for data management systems, yet it also exposes a critical bottleneck in legacy software stacks: the Write-Ahead Logging (WAL) protocol. When applied to PM, traditional disk-centric WAL mechanisms suffer from pathological **Write Amplification**, which degrades performance and reduces hardware endurance.

AG-Log proposes a novel adaptive granularity WAL system designed specifically for PM. The core of AG-Log is a lightweight dynamic decision engine that intelligently chooses between **fine-grained (delta) logging** for small updates and **coarse-grained (page) logging** for large writes on a per-operation basis.

-   For **small updates**, the system uses delta logging to record only the changes, minimizing write amplification.
-   For **large writes**, it switches to page logging for greater efficiency.

This implementation reproduces the experimental setup from the paper to compare the performance of AG-Log against traditional Page-WAL and a pure Delta-WAL across various workloads.

## The Core Concept: The Granularity Dilemma

When designing a logging system for Persistent Memory, developers face a fundamental trade-off:

1.  **Coarse-Grained (Page) Logging**:
    -   **Pros**: Simple to implement and efficient for large updates or full-page initializations.
    -   **Cons**: Suffers from pathological write amplification for small, frequent updates (e.g., modifying an account balance in an OLTP workload). For example, updating an 8-byte value might force a write of an entire 4KB page, resulting in a write amplification factor (WAF) of 512x.

2.  **Fine-Grained (Delta) Logging**:
    -   **Pros**: Directly solves the write amplification problem. The physical log size is proportional to the logical data change, maximizing performance and PM endurance for small updates.
    -   **Cons**: Each delta record requires its own metadata (e.g., transaction ID, offset, length). When a single transaction involves many small, scattered writes, this metadata overhead can become significant. For large updates, the cumulative cost of many delta records can exceed the cost of a single page log.

**AG-Log's solution** is its **dynamic decision engine**, which resolves this dilemma by choosing the optimal logging granularity at runtime based on the size of the write operation.

## Repository Structure

```
.
├── ag_log_experiment.py    # Main experiment script with all simulation and implementation logic
├── results/(results。zip)  # Directory for generated plots and CSV results after a run
│   ├── fig5_throughput.pdf
│   ├── fig6_latency.pdf
│   ├── fig7_waf.pdf
│   └── fig9_recovery.pdf
│   └── experiment_results.csv
└── README.md               # This README file
```

## Requirements

To run this experiment, you need Python 3 and the following libraries:

-   `pandas`
-   `matplotlib`
-   `seaborn`

You can easily install these dependencies using pip:

```bash
pip install pandas matplotlib seaborn
```

## How to Run the Experiment

1.  **Clone or download this repository.**

2.  **Install dependencies.**
    If you haven't already, run the pip command from the previous section.

3.  **Execute the experiment script.**
    Run the following command in your terminal:

    ```bash
    python ag_log_experiment.py
    ```

4.  **View the results.**
    The script will print the results of each experiment to the console as it runs. Upon completion:
    -   A detailed CSV file with all experimental data will be saved to `results/experiment_results.csv`.
    -   Performance comparison plots (in PDF format), corresponding to the figures in the paper, will be saved in the `results/` directory.

## Code Implementation Overview

The `ag_log_experiment.py` script contains the entire logic for the experiment, divided into several key components:

-   **`Config` Class**:
    Manages all configuration parameters for the experiment, such as page size (`PAGE_SIZE`), AG-Log's granularity threshold (`GRANULARITY_THRESHOLD`), and workload parameters.

-   **`EmulatedNVM` Class**:
    Simulates Persistent Memory using a memory-mapped file. It provides basic `read`, `write`, and `flush` operations and accurately tracks the total bytes written, which is crucial for calculating the **Write Amplification Factor (WAF)**.

-   **Logging Implementations (`PageWAL`, `DeltaWAL`, `AGLog`)**:
    -   `AbstractWAL`: An abstract base class defining a common interface for all WAL implementations.
    -   `PageWAL`: A traditional coarse-grained logger that records the entire 4KB page for every update.
    -   `DeltaWAL`: A fine-grained logger that records only the before-and-after images of the modified bytes.
    -   `AGLog`: The core implementation of the paper. When logging an update, it checks the size of the new data. If it's smaller than `GRANULARITY_THRESHOLD`, it uses delta logging; otherwise, it uses page logging.

-   **`PagedKVStore` Class**:
    A simplified page-based key-value store. All data modification operations (`put`) are first recorded using the configured WAL implementation.

-   **Workload Generators (`ycsb_workload`, `tpcc_simplified_workload`)**:
    Generates different test workloads as described in the paper:
    -   **YCSB-A (small)**: 50% reads / 50% updates with small values (64 bytes).
    -   **YCSB-B (small)**: 95% reads / 5% updates with small values (64 bytes).
    -   **YCSB-A (large)**: 50% reads / 50% updates with large values (1024 bytes).
    -   **TPC-C (Simplified)**: Simulates the core transactions of TPC-C, featuring a mix of large and small writes.

-   **Experiment Runner and Plotting**:
    The main `if __name__ == "__main__":` block orchestrates the entire experiment. It iterates through each WAL implementation and workload, collects performance metrics (throughput, latency, WAF, recovery time), and finally aggregates the results into a pandas DataFrame to generate plots.

## Interpreting the Results

After running the code, you will be able to reproduce the core findings of the paper:

-   **Write Amplification (WAF)**: For small-write workloads (YCSB-small, TPC-C), `PageWAL` exhibits a catastrophic WAF, whereas `DeltaWAL` and `AGLog` perform near-optimally.
-   **Throughput**: In write-intensive workloads, `AGLog` and `DeltaWAL` achieve significantly higher throughput than `PageWAL`. In the YCSB-large workload, `AGLog`'s performance automatically converges with that of `PageWAL`, as it correctly identifies that page logging is more efficient for large writes.
-   **Adaptivity**: The key to AG-Log's success is its ability to dynamically switch to the superior strategy for any given workload. This allows it to deliver robust, high performance across diverse scenarios, avoiding the pitfalls of any single static policy.
