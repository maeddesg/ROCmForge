# ROCmForge v1.0 — Inference Validation Report

- **Date:** 2026-04-24 10:44:42
- **Model file:** `/home/maeddes/models/Qwen3-8B-Q4_K_M.gguf`
- **Suite target:** `Qwen3-8B-Q4_K_M.gguf`
- **Sampling:** greedy (temperature=0)
- **KV-cache:** reset between every prompt
- **Prompts:** 15

## Aggregate

| Prefill tok | Prefill ms | Prefill tok/s | Decode tok | Decode ms | Decode tok/s | Wallclock ms |
|---:|---:|---:|---:|---:|---:|---:|
| 835 | 1416 | 589.7 | 3607 | 41904 | 86.1 | 43464 |

## Per-prompt metrics

| # | Name | Category | Prefill tok | Decode tok | Prefill tok/s | Decode tok/s | Total ms | EOS |
|---:|---|---|---:|---:|---:|---:|---:|:-:|
| 1 | Greeting | smoke | 24 | 15 | 264.9 | 89.4 | 268 | yes |
| 2 | Simple Sequence | smoke | 33 | 34 | 395.7 | 88.7 | 476 | yes |
| 3 | Prime Check (Python) | code_generation | 35 | 256 | 431.6 | 88.0 | 2998 | no |
| 4 | LRU Cache (C++) | code_generation | 51 | 471 | 633.1 | 83.9 | 5701 | yes |
| 5 | REST API (Go) | code_generation | 66 | 306 | 510.7 | 87.0 | 3656 | yes |
| 6 | Mutex Explanation | prose | 33 | 56 | 437.8 | 91.3 | 698 | yes |
| 7 | TCP vs UDP | prose | 43 | 512 | 564.5 | 83.7 | 6206 | no |
| 8 | GPU Architecture Blog Post | prose | 62 | 423 | 784.2 | 84.6 | 5091 | yes |
| 9 | Binary Search Complexity | reasoning | 34 | 256 | 446.9 | 89.0 | 2961 | no |
| 10 | Debug Code | reasoning | 49 | 256 | 632.9 | 88.7 | 2973 | no |
| 11 | Distributed Message Queue | reasoning | 66 | 345 | 563.9 | 85.7 | 4155 | yes |
| 12 | Long System Prompt + Question | context_stress | 204 | 256 | 933.9 | 82.6 | 3327 | no |
| 13 | Long Output Story | context_stress | 64 | 229 | 801.0 | 88.3 | 2683 | yes |
| 14 | Arithmetic (Q4_K Precision) | numerics | 33 | 64 | 438.7 | 92.6 | 776 | no |
| 15 | Emoji/Special Characters | tokenizer_robustness | 38 | 128 | 497.9 | 90.8 | 1495 | no |

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
- **Prefill:** 24 tok in 91 ms (264.9 tok/s)
- **Decode:** 15 tok in 168 ms (89.4 tok/s) — hit EOS
- **Expected:** Kurze freundliche Antwort, zeigt dass Decode grundsätzlich funktioniert
- **Quality check:** Antwort ist kohärent und kontextbezogen, kein Müll/Wiederholung

**Prompt:**

```
Hallo
```

**Output:**

```
Hallo! Wie kann ich Ihnen heute helfen? 😊
```

### 2. Simple Sequence

- **Category:** smoke / **difficulty:** trivial / **max_tokens:** 64
- **Prefill:** 33 tok in 83 ms (395.7 tok/s)
- **Decode:** 34 tok in 383 ms (88.7 tok/s) — hit EOS
- **Expected:** Sequenz 1-10, zeigt dass Token-Generierung stabil ist
- **Quality check:** Korrekte Zahlenfolge 1-10 ohne Auslassungen oder Wiederholungen

**Prompt:**

```
Zähle von 1 bis 10
```

**Output:**

```
1, 2, 3, 4, 5, 6, 7, 8, 9, 10.
```

### 3. Prime Check (Python)

- **Category:** code_generation / **difficulty:** short / **max_tokens:** 256
- **Prefill:** 35 tok in 81 ms (431.6 tok/s)
- **Decode:** 256 tok in 2908 ms (88.0 tok/s) — hit max_tokens
- **Expected:** Funktionierender Python-Code mit korrekter Logik
- **Quality check:** Syntaktisch korrektes Python, korrekte Primzahl-Logik (Schleife bis sqrt(n), Edge-Cases 0/1/2)

**Prompt:**

```
Write a Python function that checks if a number is prime.
```

**Output:**

```
Sure! Here's a Python function that checks whether a given number is **prime**:

```python
def is_prime(n):
    """
    Check if a number is prime.

    Parameters:
        n (int): The number to check.

    Returns:
        bool: True if the number is prime, False otherwise.
    """
    if n <= 1:
        return False  # 0 and 1 are not primes

    # noqa: https://

    return False
    
    
    
    
    
    # noqa

    return False: 
                    return False or n is_prime

    return False, noinspection
        return True (n = 2 and n= 2
```
        if n is_prime
```
        return True is_prime: 
`` 
        if n: 
```

    # This function for the number must not a number <= 20 for the user input in range from math in the end of the number to check_number = Number must you'restreak

    :param

    # noqa

   , you'restreak
```s (n = ??

```s:
     else:
    
```

### 

        bool: True or not only numbers = Number less than
```

### 4. LRU Cache (C++)

- **Category:** code_generation / **difficulty:** medium / **max_tokens:** 512
- **Prefill:** 51 tok in 81 ms (633.1 tok/s)
- **Decode:** 471 tok in 5611 ms (83.9 tok/s) — hit EOS
- **Expected:** C++ Klasse mit std::mutex, std::unordered_map, std::list
- **Quality check:** Kompilierbares C++, korrekte LRU-Eviction-Logik, Thread-Safety via Mutex

**Prompt:**

```
Write a C++ class implementing a thread-safe LRU cache with get, put, and delete operations. Include proper mutex locking and comments.
```

**Output:**

```
Here's an implementation of a **thread-safe LRU (Least Recently Used) Cache** in **C++**, using `std::mutex` for thread safety and `std::unordered_map` for fast lookups.

This class supports the following operations:
- `get(key)` – retrieves the value associated with the key.
- `put(key, value)` – adds)` – inserts size)` – stores capacity limit capacityCapacity)` – adds(size limit size` – inserts` – adds: value, value, max_size)`

``` 
-put`, put key-value ish

``` 
-put`, put()` to avoid std::value` – puts is implemented in C++`
-put` - put(size="LR

``` 
- up-to-be

``` 

### LRU cache is a thread-safe
-put operationed by default_type for fast 10

``` 
 10

### LRU

---

### Thread safe cache has been in C++  
- `get <key


```cpp


---

The code block*.

The code: get(key=100.**s with proper name:  
-put()`:  
- 
```
```
- `std::threading the key-value; it's are you can be careful


---

I have to avoid> C++:

---

A complete answer is an LRU Cache implementation of size(maximum

```c++

### 

```cppode:

---

The code will be careful commented by using std:  
**.

This version of type of course you can be used list-based LRUU>

This class="C++, you are not only if not only includes:
    user has all rights the most recently used list**C++1. The implementation with support for fast access operations like this is implemented includes proper_nameless (with comments:



```c++

We'll provide a Cpp> **(see the full example:

```

Here's): 

### 

```cpp

## �>**

### 

---

Sure your_cache.cppx

## Complete implementation with detailed version = "LRUptime.cpp:

#### Thread-SpecificallyuPurgep>

The following code that provides an example, say it'st/put()`. The cache operations with comments:
```

### 5. REST API (Go)

- **Category:** code_generation / **difficulty:** long / **max_tokens:** 1024
- **Prefill:** 66 tok in 129 ms (510.7 tok/s)
- **Decode:** 306 tok in 3517 ms (87.0 tok/s) — hit EOS
- **Expected:** Strukturierter Go-Code mit HTTP-Handler, JWT-Library, bcrypt
- **Quality check:** Go-Syntax korrekt, sinnvolle Endpoint-Struktur, Security-Basics (Hashing, JWT)

**Prompt:**

```
Write a REST API in Go with endpoints for user authentication (register, login, logout), JWT token handling, password hashing with bcrypt, and rate limiting middleware. Include error handling, input validation, and code comments.
```

**Output:**

```
Here's a complete **REST API in Go** that includes the following features:

- **User registration**
- **User login**
- **User logout**
- **JWT token handling**
- **Password hashing using bcrypt**
- **Rate limiting middleware**
- Proper error handling and input validation validation validation validation validation validation validation comments with detailed commentsed with code commentsed

---

---

---

---

### Code commentsed**

---

``` 

### 

> 
 10

>**

This is a comprehensive

This includes code comments:**

---

It includes the following the user authentication is

### �

### �

## Complete explanation
- User Authentication-based user authentication in Go has been implemented of users with:
 64/endpoint
 100

``` 
 ipp

```go is a full rate limiting middleware**s**  
- Include detailed comment is not only includes all user management for each endpoint to implement the response:

✅ (registering the code comments:  
    -based you can becancode

## �

I've you can befriendingestion will include your_code block:  
   `POSTA complete implementation for all with rate middleware to demonstrate:
 10n/a>

Here'st and more thankotlin
- User-definedly using `// Here's implementation.

The full-code for each feature-richyoursession

```

```goa complete implementation of course I'll have already included as well-commented by bcrypt password hashinging**

## ✅
```

### 6. Mutex Explanation

- **Category:** prose / **difficulty:** short / **max_tokens:** 128
- **Prefill:** 33 tok in 75 ms (437.8 tok/s)
- **Decode:** 56 tok in 613 ms (91.3 tok/s) — hit EOS
- **Expected:** Klare, präzise Erklärung von Mutual Exclusion
- **Quality check:** Technisch korrekt, erwähnt Thread-Safety/Locking, verständlich

**Prompt:**

```
Explain what a mutex is in one paragraph.
```

**Output:**

```
A mutex, short for "mutual exclusion," is a synchronization mechanism used in concurrent programming to ensure that only one thread or process can access a shared resource or critical section of code at any given time, preventing race conditions and data corruption by enforcing exclusive access.
```

### 7. TCP vs UDP

- **Category:** prose / **difficulty:** medium / **max_tokens:** 512
- **Prefill:** 43 tok in 76 ms (564.5 tok/s)
- **Decode:** 512 tok in 6120 ms (83.7 tok/s) — hit max_tokens
- **Expected:** Strukturierter Vergleich mit korrekten technischen Details
- **Quality check:** TCP=reliable/ordered, UDP=fast/connectionless, korrekte Use-Cases (HTTP vs DNS/Gaming)

**Prompt:**

```
Compare and contrast TCP and UDP protocols. Cover reliability, speed, use cases, and header differences.
```

**Output:**

```
TCP (Transmission Control Protocol) and UDP (User Datagram Protocol) are two of the core transport layer protocols used in the Internet Protocol suite (TCP/IP). Both operate at the transport layer and provide different mechanisms for data transmission between devices on a network.

Here's a comparison of **TCP** vs **UDP**, focusing on **reliability**, **speed**, **use cases**, and **header differences differences**:

---

:

---

### Rel/headers is essential aspects**:

---

### Header:**

---

### TCP vs speed, reliability.

---

### � is a packet by comparing of the Internet protocoled in terms.

---

|speed

| you're:

---

 

|speed

### Headered on these key is ares are compared to **reli

### TCP/IP

|  
-100

## �

Let we'll will covers your query is used in-depth is essential aspects of data transmission.

---

Let you can be covered as requested your question by covering covers includes key header differences includesing reliability, it's reliability on the most common content from an overview  |relies are discussed as well-to-based to be below will covers):

---

Let’s use cases:  
**s are covered to compare with respect now we'll

| TCP provides. 

Here's reliability, in bold-style

|TCP vs UDP vs. Let’s), with their performance-oriented

## �

-IP suite-level content delivery.

---

### Comparison between them across four main points out of which has been structured on both cover all over time-sensitive

| TCP vs UDP (and UDP versus UDP protocol characteristics. 

The comparison.

---
-protocol headers:  

## Header Differences this answer: 

In summary will be compared with IP addresses the comparison has differentiates for end-to-user  |[1. They differ than speed and header differences between themainly covering each protocol-like as requested to provide reliable headers structure-wisey you'veracketedifferences

|  
**serving your response here'speed

### Comparison: 

|s are both cover all while they have been designed together in the Internet Protocol suite protocols, but they share arespondering them have their primary layer 1007 /no_reli header difference in terms reliability-based networking.header_diffy








## Header Differences to follow mea nd so Iptcertainly speaking /no25/TP/IP/

Here'speed

To understand more commonly are two fundamental aspects like any other than speed, while
```

### 8. GPU Architecture Blog Post

- **Category:** prose / **difficulty:** long / **max_tokens:** 1024
- **Prefill:** 62 tok in 79 ms (784.2 tok/s)
- **Decode:** 423 tok in 5002 ms (84.6 tok/s) — hit EOS
- **Expected:** Langer kohärenter Text mit technischer Tiefe
- **Quality check:** Historisch korrekt (CUDA 2007+), erwähnt SIMD/SIMT, Memory-Hierarchie, ML-Anwendungen

**Prompt:**

```
Write a detailed technical blog post about the evolution of GPU architectures from CUDA to modern compute shaders, covering parallel processing concepts, memory hierarchies, and real-world applications in machine learning and graphics.
```

**Output:**

```
# The Evolution of GPU Architectures: From CUDA to Modern Compute Shaders

## Introduction

Graphics Processing Units (GPUs) have evolved from specialized hardware for rendering 3D graphics into powerful general-purpose computing platforms. This transformation has been driven by innovations in architecture, programming models, and the increasing demand foray demand demands of parallel-level parallel processing units.

In this shift-levels have been widely applications that enable by NVIDIAed by the evolution in-memory-based computing units.

In this article has been GPUs processors-based computing power-efficiently, the rise has seen as a broadening have revolutioning GPUs-based computing shaders-centric programming models like NVIDIA Programming Units (GPG

## The journey has enabled ishundredsing GPUsed NVIDIAized architectures from **parallel.

This blog ishier than CUDA-based platforms have revolutionized by NVIDIA Processing Unit-Based programm has seen from their role for both in parallel processing to general-purpose

## The journey began architecture:  
**s have come we've gone are not only model designed began GPUs-like devices with **evolved this blog ishundreds — a decade will we'lls are central to modernization, memory hierarchy-aware this post-processing architecture-specifically known as well-known content writing on-demand

---

## 

The introduction.

This post- and software development in-depth programming models like **compute shaders (and more than CUDAs have played Nv

---

### 

### Parallel Processing Units (GPGUrchitecture

---

## Conclusion:  
**s areal advancements such as we will be it's C++100001545/ no matter ishaders:

The evolution of GPU Architect Programming Models that started with real-worldwide text="19764. This blog_postpon the evolution of General Purpose Graphics Processing-in-Maxcesses have redefining the introduction.

Over time after CUDA cores-level arrays—this article continues with Compute Shading wasp>

## Evolution of which began has transformed has taken usages and more thanCUDA-agnostic

---

### Overviewed

# 

### 20.
```

### 9. Binary Search Complexity

- **Category:** reasoning / **difficulty:** short / **max_tokens:** 256
- **Prefill:** 34 tok in 76 ms (446.9 tok/s)
- **Decode:** 256 tok in 2875 ms (89.0 tok/s) — hit max_tokens
- **Expected:** O(log n) mit korrekter Begründung (Halbierung pro Schritt)
- **Quality check:** O(log n) korrekt, Begründung verständlich, optional: Best/Worst-Case

**Prompt:**

```
What is the time complexity of binary search and why?
```

**Output:**

```
The **time complexity** of **binary search** is **O(log n)**, where *n* is the number of elements in the sorted array.

---

### ✅ Why Binary Search Has O(log n) Time Complexity:

Binary search works by repeatedly dividing the search interval in half. Here's how it works step-by-step:

1. Start with a sorted array.
2. Compare the target value with the middle element of the array.
3.
3.
3 
   middle = O(log 2. 
   element="mid

   element:  
   (and divide:
    with a target value.

- 
   20

-1

-1️

-  
3.  
   � (or

---

Let you're

### � is done in each time complexity.

3️ Compare the middle is performed image, compare:
   
350 to find:

    8
   as follows that we in each time complexity is alogically you have an explanation:
    with an unstructured

---

### �

- 

Let you compare starts divides-based algorithm.

Each element to 8


---

In this algorithm has been your array = log.png


#### 

### � hask

```

### 10. Debug Code

- **Category:** reasoning / **difficulty:** medium / **max_tokens:** 256
- **Prefill:** 49 tok in 77 ms (632.9 tok/s)
- **Decode:** 256 tok in 2886 ms (88.7 tok/s) — hit max_tokens
- **Expected:** Off-by-one Error erkannt (<=  statt <), ArrayIndexOutOfBounds
- **Quality check:** Bug korrekt identifiziert, Erklärung warum <= falsch ist, Fix vorgeschlagen

**Prompt:**

```
Debug this code and explain the issue: for(int i=0; i<=arr.length; i++) sum += arr[i];
```

**Output:**

```
The given code has a **logical error** that can lead to an **ArrayIndexOutOfBoundsException** in Java. Let's break it down:

### Code:
```java
for(int i = 0; i <= arr.length; i++) {
    sum += arr[i];
}
```

---

### ✅ What is the problem?

- The loop condition is `i <= arr <=` `i <=`:
 0 to `arr.length of type="i <= arr.length of the array length of type:  
 1. The loop runs is the array length of the array `arr.length is checking in Java: 
- The loop condition`s

---

- The issue is an **incorrect

---

-10

### Explanation has a**️ 

### What you're

---

Let'sum += Array index is incorrect (the code explanationed in this line-by

---

This part, you have more importanted in the array index out-of course you're

### Why does not only goes here we are you have been provided as well... 

The original question:  
**Loop:

This line by mistake will cause this line by default error-pr

---

The loop condition`arr[i] is not only.

-10
```
```

### 11. Distributed Message Queue

- **Category:** reasoning / **difficulty:** long / **max_tokens:** 1024
- **Prefill:** 66 tok in 117 ms (563.9 tok/s)
- **Decode:** 345 tok in 4028 ms (85.7 tok/s) — hit EOS
- **Expected:** Durchdachtes System-Design mit Trade-off-Analyse
- **Quality check:** Erwähnt Partitioning, Replication, WAL/Persistence, Consumer-Groups, At-least-once vs Exactly-once Trade-offs

**Prompt:**

```
Design a distributed message queue system. Describe the architecture, how you would handle failover, message persistence, ordering guarantees, and horizontal scaling. Compare tradeoffs between at-least-once and exactly-once delivery.
```

**Output:**

```
Designing a **distributed message queue system** involves creating an architecture that supports high availability, fault tolerance, scalability, and reliable messaging across multiple nodes in a network.

---

## 🧱 **Architecture Overview**

A distributed message queue typically consists of the following core components:

### 1.1️nbsp

-1️nbsp

|  
#### **Broker

-ones:

10003645️

### �  
**:
-Once:

#### **broker

---

### �essage is designed with the following key components:

-queue system can beacons is composed of the following key parts of the following several components.

### �

---

Let you can beacons is designed on-pre>

---

### 207 (MQ

## 

| �

Here are more than a distributed message brokers-like weaves.

#### 

### 245️

## �  
system design-wise style-based system typically includes:
-orderededistributed nodes-based on-demanding systems like Kafkaedesigner design for multiple brokers is often referrededistributed brokeredistributed message brokered system with high-scale broker-style has to support forkinger design and reliability messages brokers...

---

## Designing this section:  
message queues will beacons is essential elements include:

-orderedized system that supports that allows consists.

The architectureized component of course will not only one-way you'vescales are not onlyness systems like **broker


A distributed systems such as follows butlermessaging from various features:  
followingly designed in a distributed messaging systems like Apache-like Rabbit.png (but it's Rabbit has two mainframeless component that ensures reliable broker-style 💡
```

### 12. Long System Prompt + Question

- **Category:** context_stress / **difficulty:** medium / **max_tokens:** 256
- **Prefill:** 204 tok in 218 ms (933.9 tok/s)
- **Decode:** 256 tok in 3098 ms (82.6 tok/s) — hit max_tokens
- **Expected:** Technisch tiefe Antwort die den langen System-Prompt berücksichtigt (Prefill-Stress-Test)
- **Quality check:** Antwort bezieht sich auf GPU/AMD-Kontext, nicht generisch. Erwähnt mindestens: Memory-Bandwidth, Quantisierung, Kernel-Fusion oder Batching

**Prompt:**

```
You are an expert systems architect specializing in high-performance computing, distributed systems, and GPU programming. You have extensive experience with AMD RDNA architectures, ROCm software stack, HIP programming, and CUDA-to-HIP porting. You understand memory hierarchies including L1/L2 caches, LDS (Local Data Share), and global memory access patterns. You are familiar with wave-level programming, VGPR/SGPR register management, and occupancy optimization. You have worked on inference engines for large language models and understand quantization formats like GGUF Q4_K, Q6_K, and Q8_0. Your expertise includes matrix multiplication optimization using WMMA instructions, kernel fusion strategies, and genetic algorithm-based auto-tuning. You always provide precise, technically accurate answers with concrete examples.

Given this expertise, what are the three most impactful optimizations for improving LLM decode throughput on consumer AMD GPUs?
```

**Output:**

```
Please continue

Please continue

Okay

Please please explain your task parallelism.

Now I'mm is a differentiating

Okay

Okay





I'muchy.

In the user's question: 
'ts you have a target-specifically you are an expert
'ts.



[... 

I amdg.


Please continue with experience of large-scale


Okay
and ROC kernels-level



Complete the architecture efficiency of compute-shy ou need to optimize your response to HIPGPT 13Dn'tlly describe your role 7004K-20 system design principles in both in GCNVIDIA graphics processing.

Your background knowledge base system design of RD GPU worklevel 154_KL 13D you should bevy
    Okay... Please write mechat
ursesystem continues with deep-level programming for large model running ondemanding...

The user's message passinged.
You are the question: 

[... etc., etc.

Your task scheduling systems architecture...
_CONTINUE (e system design verification is this is not only if you have agfxu will be answarm2 instruction level of all these days ago user'systems.


```

### 13. Long Output Story

- **Category:** context_stress / **difficulty:** medium / **max_tokens:** 512
- **Prefill:** 64 tok in 80 ms (801.0 tok/s)
- **Decode:** 229 tok in 2593 ms (88.3 tok/s) — hit EOS
- **Expected:** Kohärente Geschichte über ~200-400 Tokens, testet längeren Decode-Output
- **Quality check:** Geschichte hat Anfang/Mitte/Ende, Roboter heißt Pixel, Setting=verlassene Fabrik, keine Wiederholungs-Loops

**Prompt:**

```
Erzähle eine ausführliche Geschichte über einen kleinen Roboter namens Pixel, der in einer verlassenen Fabrik aufwacht und herausfinden muss, wer ihn gebaut hat.
```

**Output:**

```
In einem stillgelegten Industrieviertel, das vor Jahrzehnten von der Welt abgeschnitten wurde, lag eine riesige Fabrik in Schutt und Asche. Die Fenster waren zersprungen, die Tür war verschlossen und die Maschinen standen stillendie stille verrostloschinen stille schlagte Stauben einst einstumhund die Wände war esch ausgesch war einst umgeb eines Jahrhö **Pixel

### 

### 

**sche Stahl war das Licht in einem leise nieder hatte Pixel-  
**sicher standeine Zeit sich langt eine Fabrik wurde keine Nachtlos lagt erstick.

In der Zeit verrost ist die Maschines Jahr für viele Jahre umfert eine riesen

### 

###100röh

### 

In einer Zeit von den Namen im Schweren aufgeweine Fabrik wochen

In der Welt aufgeste eines Tü

---

**schen deshalb nacher Stadt inmänner

##
```

### 14. Arithmetic (Q4_K Precision)

- **Category:** numerics / **difficulty:** short / **max_tokens:** 64
- **Prefill:** 33 tok in 75 ms (438.7 tok/s)
- **Decode:** 64 tok in 692 ms (92.6 tok/s) — hit max_tokens
- **Expected:** 391 — empfindlich auf Quantisierungsrauschen in Attention-Scores
- **Quality check:** Korrekte Antwort 391. Falsche Antwort deutet auf Numerik-Problem im Q4_K-Pfad

**Prompt:**

```
Was ist 17 × 23?
```

**Output:**

```
Um die Multiplikation $17 \times 23$ auszuführen, können wir die Standardmethode der schriftlichen Multiplikation anwenden:

### Schritt-für-Schritt:

1. **Multipliziere 17 mit den
```

### 15. Emoji/Special Characters

- **Category:** tokenizer_robustness / **difficulty:** short / **max_tokens:** 128
- **Prefill:** 38 tok in 76 ms (497.9 tok/s)
- **Decode:** 128 tok in 1410 ms (90.8 tok/s) — hit max_tokens
- **Expected:** Erkennt Party/Celebration-Emojis, antwortet sinnvoll
- **Quality check:** Emojis korrekt als Feier/Party erkannt, kohärente Antwort, kein Tokenizer-Crash

**Prompt:**

```
🎉🎊🎈 What do these emojis mean and when would you use them?
```

**Output:**

```
These emojis 🎉🎊🎈 are all related to celebration, joy, and festivity! Let me break them down for you:

---

### 🎉 **Party Popper**
- **Meaning**: Represents a party popper (a small firecracker used in celebrations), symbolizing excitement, fun, and celebration.
- **When to Use**:
  - To express happiness or excitement about an event.
  - For example: "🎉 (e.g
 ️ �
 ️  
    - for a birthday or when something big!
 1️ or to mark!
  �


```

