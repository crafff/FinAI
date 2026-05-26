# Multi-Agent LLM System for 10-K-Based One-Week Stock Prediction

**Group 4** · Ruitao Zhou, Lauren McDonald, Gustavo Mercier

A collaborative multi-agent LLM system that uses companies' FY2025 10-K annual reports to predict the direction (buy / not-buy) and one-week target price of their stock following the filing release. The system extends Han et al. (ICAIF '24) with a **task-aware coopetition pipeline**: agents cooperate on information-gathering subtasks and compete on judgment subtasks, with a dedicated red-team agent and a bounded rebuttal–revision loop.

## Project Documents

- 📄 [Full Specification (Problem Definition & System Design)](https://docs.google.com/document/d/1VocIfIF5fPdbh5KMorKfWYynwCTsxhQMe2RptiX_hks/edit?usp=sharing)
- 📋 [Task Breakdown (Independent Subtasks)](https://docs.google.com/document/d/1fqnrskBB9yV1Fxjw3reCdHsairjY_vRwnn5q5ubq44s/edit?usp=sharing)

## Overview

Given each company's FY2025 10-K, the system runs a four-stage pipeline:

1. **Subtask analysis** — Fundamental analyst and Sentiment scout (cooperative); Risk analysis via two opposing agents (qualitative vs. quantitative) in a three-phase protocol.
2. **Leader aggregation** — Free-judgment initial prediction with a mandatory rationale.
3. **Red-team rebuttal** — A dedicated Evaluation agent attacks the prediction; multi-round, capped.
4. **Final prediction** — Buy/not-buy + one-week target price.

Prediction protocol: baseline = the T₀ close (first tradable day after release), target = the close on the 5th trading day after T₀. All agent-visible information is cut off at the T₀ close to prevent look-ahead leakage.

## Tech Stack

- **Orchestration:** LangGraph
- **Data:** SEC EDGAR (10-K), yfinance (prices), FMP (financials), FinnHub (news), Reddit/PRAW (social)
- **Retrieval:** RAG over chunked 10-K text

## Evaluation

Three-way ablation compares a single-agent baseline, a paper-style aggregation baseline, and the full system, measuring directional accuracy (with confidence intervals), target-price percentage error, and correlated-error rate.

## Repository Structure

```
.
├── data/            # EDGAR, price, news, social retrieval + T0 logic
├── agents/          # subtask, leader, and red-team agents
├── orchestration/   # LangGraph state graph
├── evaluation/      # metrics, statistical tests, ablation experiments
└── README.md
```

> **Note:** This project is for academic research. It does not constitute investment advice.
