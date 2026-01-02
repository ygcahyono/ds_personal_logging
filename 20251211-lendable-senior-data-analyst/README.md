# Personal Loan Success Analysis

> Identifying key factors that determine personal loan application success to improve approval rates and reduce risk.

## Problem

A personal loan company needs to understand what drives successful loan applications. Currently, approval decisions lack clear data-driven insights, leading to potential missed opportunities and unclear risk factors.

**Goal**: Build an interpretable model to identify which applicant and loan characteristics predict success.

## Solution

- **Model**: Logistic Regression (interpretable for business stakeholders)
- **Accuracy**: XX%
- **Key Findings**: 
  - Employment status is the strongest predictor
  - 4-year loan terms show higher success rates
  - Credit history (payment discipline, defaults, recent credit inquiries) significantly impacts approval

## Impact

- **Product**: Prioritise features that matter most (employment verification, loan term flexibility)
- **Marketing**: Target employed applicants with 4-year term products
- **Risk**: Quantify how credit history factors affect success probability

## Flow

```mermaid
graph TD
    %% Node Definitions
    Start[(Data Loading)] --> Schema[Schema Review & Understanding]
    Schema --> Transform[Data Transformation]
    
    %% The Cleaning Loop
    Transform --> EDA[EDA + Visualisation]
    EDA -- "Data-Cleaning Loop" --> Transform
    
    %% Transition to Modeling
    EDA --> Ready[Clean Table + Inference Ready]
    
    %% Modeling Phase Subgraph
    subgraph Modeling_Phase [Modeling Phase]
        direction LR
        Assump[Model's Assumption Transformation] --> Run[Running the Model]
        Run --> Interp[Interpretation]
    end
    
    Ready --> Assump

    %% Styling
    style Start fill:#f9f,stroke:#333,stroke-width:2px
    style Ready fill:#00d2ff,stroke:#333,stroke-width:2px,color:#fff
    style Modeling_Phase fill:#f5f5f5,stroke:#666,stroke-dasharray: 5 5
    style Interp font-weight:bold,fill:#90ee90
```


- `yogi_takehome-test_fintech01_senior-data-analyst.ipynb` - Main analysis
- `data/` - Raw data

---
Yogi Cahyono | [15 December 2025]