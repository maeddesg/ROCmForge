# ROCmForge v1.0 — Inference Validation Report

- **Date:** 2026-04-24 14:31:55
- **Model file:** `/home/maeddes/models/Meta-Llama-3.1-8B-Instruct-Q4_K_M.gguf`
- **Suite target:** `Qwen3-8B-Q4_K_M.gguf`
- **Sampling:** greedy (temperature=0)
- **KV-cache:** reset between every prompt
- **Prompts:** 15

## Aggregate

| Prefill tok | Prefill ms | Prefill tok/s | Decode tok | Decode ms | Decode tok/s | Wallclock ms |
|---:|---:|---:|---:|---:|---:|---:|
| 806 | 1194 | 675.1 | 606 | 5750 | 105.4 | 7105 |

## Per-prompt metrics

| # | Name | Category | Prefill tok | Decode tok | Prefill tok/s | Decode tok/s | Total ms | EOS |
|---:|---|---|---:|---:|---:|---:|---:|:-:|
| 1 | Greeting | smoke | 22 | 18 | 303.1 | 107.6 | 251 | yes |
| 2 | Simple Sequence | smoke | 30 | 22 | 423.6 | 107.1 | 287 | yes |
| 3 | Prime Check (Python) | code_generation | 33 | 13 | 464.1 | 107.3 | 203 | yes |
| 4 | LRU Cache (C++) | code_generation | 49 | 31 | 673.3 | 107.2 | 373 | yes |
| 5 | REST API (Go) | code_generation | 64 | 42 | 862.2 | 106.3 | 480 | yes |
| 6 | Mutex Explanation | prose | 31 | 49 | 446.0 | 107.5 | 536 | yes |
| 7 | TCP vs UDP | prose | 41 | 16 | 580.6 | 107.9 | 230 | yes |
| 8 | GPU Architecture Blog Post | prose | 60 | 12 | 825.3 | 107.2 | 195 | yes |
| 9 | Binary Search Complexity | reasoning | 32 | 33 | 459.4 | 108.1 | 385 | yes |
| 10 | Debug Code | reasoning | 47 | 21 | 656.5 | 107.3 | 278 | yes |
| 11 | Distributed Message Queue | reasoning | 64 | 232 | 870.6 | 104.0 | 2314 | yes |
| 12 | Long System Prompt + Question | context_stress | 200 | 89 | 1026.2 | 102.8 | 1072 | yes |
| 13 | Long Output Story | context_stress | 62 | 5 | 860.9 | 106.8 | 130 | yes |
| 14 | Arithmetic (Q4_K Precision) | numerics | 29 | 12 | 424.5 | 107.9 | 190 | yes |
| 15 | Emoji/Special Characters | tokenizer_robustness | 42 | 11 | 603.9 | 108.2 | 182 | yes |

## Human evaluation (fill in manually)

Rating scale: **korrekt** / **teilweise** / **falsch** / **müll**.
Pass threshold: 12/15 korrekt for Phase-1 acceptance.

| # | Name | Rating | Notes |
|---:|---|:-:|---|
| 1 | Greeting |   |   |
| 2 | Simple Sequence |   |   |
| 3 | Prime Check (Python) |   |   |
| 4 | LRU Cache (C++) |   |   |
| 5 | REST API (Go) |   |   |
| 6 | Mutex Explanation |   |   |
| 7 | TCP vs UDP |   |   |
| 8 | GPU Architecture Blog Post |   |   |
| 9 | Binary Search Complexity |   |   |
| 10 | Debug Code |   |   |
| 11 | Distributed Message Queue |   |   |
| 12 | Long System Prompt + Question |   |   |
| 13 | Long Output Story |   |   |
| 14 | Arithmetic (Q4_K Precision) |   |   |
| 15 | Emoji/Special Characters |   |   |

## Outputs

### 1. Greeting

- **Category:** smoke / **difficulty:** trivial / **max_tokens:** 64
- **Prefill:** 22 tok in 73 ms (303.1 tok/s)
- **Decode:** 18 tok in 167 ms (107.6 tok/s) — hit EOS
- **Expected:** Kurze freundliche Antwort, zeigt dass Decode grundsätzlich funktioniert
- **Quality check:** Antwort ist kohärent und kontextbezogen, kein Müll/Wiederholung

**Prompt:**

```
Hallo
```

**Output:**

```
Hello! How can I assist you with your day? How can I help you today?
```

### 2. Simple Sequence

- **Category:** smoke / **difficulty:** trivial / **max_tokens:** 64
- **Prefill:** 30 tok in 71 ms (423.6 tok/s)
- **Decode:** 22 tok in 205 ms (107.1 tok/s) — hit EOS
- **Expected:** Sequenz 1-10, zeigt dass Token-Generierung stabil ist
- **Quality check:** Korrekte Zahlenfolge 1-10 ohne Auslassungen oder Wiederholungen

**Prompt:**

```
Zähle von 1 bis 10
```

**Output:**

```
It looks like you want to chat with me? What can I can help you with any question or topic?
```

### 3. Prime Check (Python)

- **Category:** code_generation / **difficulty:** short / **max_tokens:** 256
- **Prefill:** 33 tok in 71 ms (464.1 tok/s)
- **Decode:** 13 tok in 121 ms (107.3 tok/s) — hit EOS
- **Expected:** Funktionierender Python-Code mit korrekter Logik
- **Quality check:** Syntaktisch korrektes Python, korrekte Primzahl-Logik (Schleife bis sqrt(n), Edge-Cases 0/1/2)

**Prompt:**

```
Write a Python function that checks if a number is prime.
```

**Output:**

```
Here's code to check if the input number is prime or not
```

### 4. LRU Cache (C++)

- **Category:** code_generation / **difficulty:** medium / **max_tokens:** 512
- **Prefill:** 49 tok in 73 ms (673.3 tok/s)
- **Decode:** 31 tok in 289 ms (107.2 tok/s) — hit EOS
- **Expected:** C++ Klasse mit std::mutex, std::unordered_map, std::list
- **Quality check:** Kompilierbares C++, korrekte LRU-Eviction-Logik, Thread-Safety via Mutex

**Prompt:**

```
Write a C++ class implementing a thread-safe LRU cache with get, put, and delete operations. Include proper mutex locking and comments.
```

**Output:**

```
Here's an example of how to implement the following code for the `lock` lock mechanism using `lock` lock in Python to protect the `thread`
```

### 5. REST API (Go)

- **Category:** code_generation / **difficulty:** long / **max_tokens:** 1024
- **Prefill:** 64 tok in 74 ms (862.2 tok/s)
- **Decode:** 42 tok in 395 ms (106.3 tok/s) — hit EOS
- **Expected:** Strukturierter Go-Code mit HTTP-Handler, JWT-Library, bcrypt
- **Quality check:** Go-Syntax korrekt, sinnvolle Endpoint-Struktur, Security-Basics (Hashing, JWT)

**Prompt:**

```
Write a REST API in Go with endpoints for user authentication (register, login, logout), JWT token handling, password hashing with bcrypt, and rate limiting middleware. Include error handling, input validation, and code comments.
```

**Output:**

```
## Step 1: 

Let's endpoint to handle errors 

```bash 

Here is a simple example of the user data from the user data to be used to store data from the database using SQL database.
```

### 6. Mutex Explanation

- **Category:** prose / **difficulty:** short / **max_tokens:** 128
- **Prefill:** 31 tok in 70 ms (446.0 tok/s)
- **Decode:** 49 tok in 456 ms (107.5 tok/s) — hit EOS
- **Expected:** Klare, präzise Erklärung von Mutual Exclusion
- **Quality check:** Technisch korrekt, erwähnt Thread-Safety/Locking, verständlich

**Prompt:**

```
Explain what a mutex is in one paragraph.
```

**Output:**

```
In computer science, a mutex (short for mutual exclusion lock is a synchronization primitive that prevents two threads from accessing the same resource sharing of shared resource that can be used to access to the same resource that can be accessed by multiple threads or processes.
```

### 7. TCP vs UDP

- **Category:** prose / **difficulty:** medium / **max_tokens:** 512
- **Prefill:** 41 tok in 71 ms (580.6 tok/s)
- **Decode:** 16 tok in 148 ms (107.9 tok/s) — hit EOS
- **Expected:** Strukturierter Vergleich mit korrekten technischen Details
- **Quality check:** TCP=reliable/ordered, UDP=fast/connectionless, korrekte Use-Cases (HTTP vs DNS/Gaming)

**Prompt:**

```
Compare and contrast TCP and UDP protocols. Cover reliability, speed, use cases, and header differences.
```

**Output:**

```
TCP/IP (Transmission Control Protocol vs UDP (User Datagram Protocol (TCP/IP)
```

### 8. GPU Architecture Blog Post

- **Category:** prose / **difficulty:** long / **max_tokens:** 1024
- **Prefill:** 60 tok in 73 ms (825.3 tok/s)
- **Decode:** 12 tok in 112 ms (107.2 tok/s) — hit EOS
- **Expected:** Langer kohärenter Text mit technischer Tiefe
- **Quality check:** Historisch korrekt (CUDA 2007+), erwähnt SIMD/SIMT, Memory-Hierarchie, ML-Anwendungen

**Prompt:**

```
Write a detailed technical blog post about the evolution of GPU architectures from CUDA to modern compute shaders, covering parallel processing concepts, memory hierarchies, and real-world applications in machine learning and graphics.
```

**Output:**

```
The term "GPU" (Graphics Processing Units (GPU)
```

### 9. Binary Search Complexity

- **Category:** reasoning / **difficulty:** short / **max_tokens:** 256
- **Prefill:** 32 tok in 70 ms (459.4 tok/s)
- **Decode:** 33 tok in 305 ms (108.1 tok/s) — hit EOS
- **Expected:** O(log n) mit korrekter Begründung (Halbierung pro Schritt)
- **Quality check:** O(log n) korrekt, Begründung verständlich, optional: Best/Worst-Case

**Prompt:**

```
What is the time complexity of binary search and why?
```

**Output:**

```
The complexity of an algorithm is a measure of the time complexity of an algorithm? 

The time complexity of an algorithm is a measure that measures the number of algorithms.
```

### 10. Debug Code

- **Category:** reasoning / **difficulty:** medium / **max_tokens:** 256
- **Prefill:** 47 tok in 72 ms (656.5 tok/s)
- **Decode:** 21 tok in 196 ms (107.3 tok/s) — hit EOS
- **Expected:** Off-by-one Error erkannt (<=  statt <), ArrayIndexOutOfBounds
- **Quality check:** Bug korrekt identifiziert, Erklärung warum <= falsch ist, Fix vorgeschlagen

**Prompt:**

```
Debug this code and explain the issue: for(int i=0; i<=arr.length; i++) sum += arr[i];
```

**Output:**

```
It looks like you have a bug in the code snippet of code that is not working with a variable.
```

### 11. Distributed Message Queue

- **Category:** reasoning / **difficulty:** long / **max_tokens:** 1024
- **Prefill:** 64 tok in 74 ms (870.6 tok/s)
- **Decode:** 232 tok in 2230 ms (104.0 tok/s) — hit EOS
- **Expected:** Durchdachtes System-Design mit Trade-off-Analyse
- **Quality check:** Erwähnt Partitioning, Replication, WAL/Persistence, Consumer-Groups, At-least-once vs Exactly-once Trade-offs

**Prompt:**

```
Design a distributed message queue system. Describe the architecture, how you would handle failover, message persistence, ordering guarantees, and horizontal scaling. Compare tradeoffs between at-least-once and exactly-once delivery.
```

**Output:**

```
In this is a distributed system that allows for the system to be used to be used to handle the data in a distributed database with multiple nodes in the same as an example of multiple nodes in a single node that can be used by using multiple nodes. The node is not just one of the following: 

This is not just one of nodes and also known as follows a single node. 

The system for an example, but also known as well as well-known node in this is called "node. This is not just one of the message queueing a distributed systems like an order to ensure that can be designed for each node, and how it's message handling messages are designed to ensure that you have been implemented by using multiple nodes are some of the same as well-known as well-known as an object-oriented approachable to send messages from the data storage systems with multiple nodes in which is not (or the networked by providing access to store data storeaafter, but also has (and then use (also known as part of nodes are stored on-preventing on-preventing multiple databases and then use of all messages are more than 1
The architecture.
```

### 12. Long System Prompt + Question

- **Category:** context_stress / **difficulty:** medium / **max_tokens:** 256
- **Prefill:** 200 tok in 195 ms (1026.2 tok/s)
- **Decode:** 89 tok in 866 ms (102.8 tok/s) — hit EOS
- **Expected:** Technisch tiefe Antwort die den langen System-Prompt berücksichtigt (Prefill-Stress-Test)
- **Quality check:** Antwort bezieht sich auf GPU/AMD-Kontext, nicht generisch. Erwähnt mindestens: Memory-Bandwidth, Quantisierung, Kernel-Fusion oder Batching

**Prompt:**

```
You are an expert systems architect specializing in high-performance computing, distributed systems, and GPU programming. You have extensive experience with AMD RDNA architectures, ROCm software stack, HIP programming, and CUDA-to-HIP porting. You understand memory hierarchies including L1/L2 caches, LDS (Local Data Share), and global memory access patterns. You are familiar with wave-level programming, VGPR/SGPR register management, and occupancy optimization. You have worked on inference engines for large language models and understand quantization formats like GGUF Q4_K, Q6_K, and Q8_0. Your expertise includes matrix multiplication optimization using WMMA instructions, kernel fusion strategies, and genetic algorithm-based auto-tuning. You always provide precise, technically accurate answers with concrete examples.

Given this expertise, what are the three most impactful optimizations for improving LLM decode throughput on consumer AMD GPUs?
```

**Output:**

```
The GPU architecture of the GPU acceleration of the following a new technologies such as well as a new to be used to be able to be used to be done by the following the memory access to create a few of the same as well as wellaaftering the memory cache memory caching of data structures that is not just-including a new technologies such as well (GPU architecture of your code generation of data centering the following an object-oriented architectures.
```

### 13. Long Output Story

- **Category:** context_stress / **difficulty:** medium / **max_tokens:** 512
- **Prefill:** 62 tok in 72 ms (860.9 tok/s)
- **Decode:** 5 tok in 47 ms (106.8 tok/s) — hit EOS
- **Expected:** Kohärente Geschichte über ~200-400 Tokens, testet längeren Decode-Output
- **Quality check:** Geschichte hat Anfang/Mitte/Ende, Roboter heißt Pixel, Setting=verlassene Fabrik, keine Wiederholungs-Loops

**Prompt:**

```
Erzähle eine ausführliche Geschichte über einen kleinen Roboter namens Pixel, der in einer verlassenen Fabrik aufwacht und herausfinden muss, wer ihn gebaut hat.
```

**Output:**

```
Ich bin einiger.
```

### 14. Arithmetic (Q4_K Precision)

- **Category:** numerics / **difficulty:** short / **max_tokens:** 64
- **Prefill:** 29 tok in 68 ms (424.5 tok/s)
- **Decode:** 12 tok in 111 ms (107.9 tok/s) — hit EOS
- **Expected:** 391 — empfindlich auf Quantisierungsrauschen in Attention-Scores
- **Quality check:** Korrekte Antwort 391. Falsche Antwort deutet auf Numerik-Problem im Q4_K-Pfad

**Prompt:**

```
Was ist 17 × 23?
```

**Output:**

```
It seems like you want to talk to me about something?
```

### 15. Emoji/Special Characters

- **Category:** tokenizer_robustness / **difficulty:** short / **max_tokens:** 128
- **Prefill:** 42 tok in 70 ms (603.9 tok/s)
- **Decode:** 11 tok in 102 ms (108.2 tok/s) — hit EOS
- **Expected:** Erkennt Party/Celebration-Emojis, antwortet sinnvoll
- **Quality check:** Emojis korrekt als Feier/Party erkannt, kohärente Antwort, kein Tokenizer-Crash

**Prompt:**

```
🎉🎊🎈 What do these emojis mean and when would you use them?
```

**Output:**

```
It looks like you can help me with your question?
```

