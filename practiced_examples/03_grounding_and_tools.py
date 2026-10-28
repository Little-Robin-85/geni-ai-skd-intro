from aiohttp import web_fileresponse
from google.genai._gaos.types.interactions import thoughtsummarycontent
from google import genai
from google.genai import types
from dotenv import load_dotenv

load_dotenv()
client=genai.Client()

# --- PART A: Deep Reasoning & Long Decode with Gemini Flash 3.8 ---
def solve_complex_architecture(prompt: str):
    response = client.models.generate_content(
        model="gemini-4-argon",
        contents=prompt,
        config=types.GenerateContentConfig(
            max_output_tokens=65536,
            #automatic_continuation=True {default}
            thinking_config=types.ThinkingConfig(
                thinking_level="medium",
                include_thoughts=True
            ),
        ),
    )


    for part in response.candidates[0].content.parts:
        if getattr(part, "thought", False):
            print(f"[INTERNAL THOUGHT PROCESS]:\n{part.text}\n{'-'*40}")
        else:
            print(f"[FINAL ANSWER]:\n{part.text}")


    # Inspect token consumption (including thoughts_token_count)
    usage = response.usage_metadata
    print(f"\nTokens -> Prompt: {usage.prompt_token_count} | "
            f"Thoughts: {usage.thoughts_token_count} | "
            f"Output: {usage.candidates_token_count}")


# --- PART B: Server-Side Multi-Turn State (Recommended for New Projects) ---
def stateful_interaction_demo():
    # Turn 1
    turn1 = client.interactions.create(
        model="gemini-3.5-flash-lite",
        input="We are designing a FastAPI microservice using PostgreSQL and Redis."
    )
    print("Turn 1:", turn1.output_text[:150], "...\n")


 # Turn 2: No need to resend history! Just pass previous_interaction_id
    turn2 = client.interactions.create(
        model = "gemini-3.5-flash-lite",
        previous_interaction_id=turn1.id,
        input = "What SQLAlchemy connection pooling settings should we use for the stack I just mentioned?"
    )
    print("Turn 2:", turn2.output_text)


# --- PART C: Client-Side Chat Helper (When Using generateContent) ---
def client_side_chat_demo():
    chat = client.chats.create(model="gemini-3.5-flash-lite")

    r1 = chat.send_message("I have 3 servers, each with 8 GPUs.")
    r2 = chat.send_message("If 2 GPUs fail in total, how many working GPUs remain?")
    print("Chat Answer:", r2.text)

    # You can inspect or serialize the client-side history at any time
    for message in chat.get_history():
        print(f"[{message.role}]: {message.parts[0].text[:160]}...")


#region solve_complex_architecture()
#solve_complex_architecture("if sophie as 14 sweats but gives 3 to Alice, none to john, and five to peter, how many does john and sophie have")
"""[INTERNAL THOUGHT PROCESS]:
**Analyzing the Sweet Distribution Problem**

Okay, so the goal is to figure out how many sweets John and Sophie each have, and then combined. First, I need to break down the information given. I'm focusing on the state of sweets and how they change.

I see that Sophie begins with 14 sweets. She then gives some away: 3 to Alice, nothing to John, and 5 to Peter.

Now, let's do the math for Sophie. She started with 14, and gave away a total of 3 + 0 + 5 = 8 sweets. That leaves Sophie with 14 - 8 = 6 sweets.

Next, John. The problem *explicitly* states that Sophie gave "none to John." This means John received 0 sweets from this exchange, and no prior amount is mentioned. So John has 0 sweets.

Finally, the question asks for both individual amounts and the total. Sophie has 6, and John has 0. Together, they have 6 + 0 = 6 sweets. Therefore, I will respond by stating each total and a combined total as requested.



----------------------------------------
[FINAL ANSWER]:
* **Sophie has 6.** (She started with 14 and gave away 3 + 5 = 8, leaving her with $14 - 8 = 6$).
* **John has 0.** (Sophie gave him none, and no starting amount was mentioned for him).

Together, they have **6**.

Tokens -> Prompt: 33 | Thoughts: 397 | Output: 70"""
#endregion

#region stateful_interaction_demo()
"""
stateful_interaction_demo()
Turn 1: That sounds like a solid, modern, and high-performance tech stack. FastAPI (Python), PostgreSQL (reliable relational storage), and Redis (caching, rat ...

Turn 2: Configuring SQLAlchemy connection pooling for a **FastAPI + PostgreSQL** microservice requires careful tuning. Because FastAPI is asynchronous and handles concurrent requests via an event loop, while traditional SQLAlchemy (`psycopg2`) is synchronous, getting the pool size wrong can lead to either starved database connections or exhausted PostgreSQL process limits.

Here are the recommended settings and best practices for your stack.

---

### 1. The Recommended Configuration (Synchronous Driver: `psycopg2`)

If you are using the standard synchronous PostgreSQL driver (`psycopg2` or `psycopg[binary]`), configure your engine like this:

```python
from sqlalchemy import create_engine
from sqlalchemy.orm import sessionmaker

DATABASE_URL = "postgresql://user:password@localhost:5432/dbname"

engine = create_engine(
    DATABASE_URL,
    # 1. Pool Size & Max Overflow
    pool_size=5,  # Number of permanent connections to keep open
    max_overflow=10,  # Max additional connections to create during high load
    # 2. Timeouts
    pool_timeout=30,  # Seconds to wait for a connection from the pool before throwing an error
    # 3. Recycling & Pre-ping (Crucial for microservices/cloud DBs)
    pool_recycle=1800,  # Recycle connections after 30 minutes to prevent stale connections
    pool_pre_ping=True,  # Test connection health before using it (avoids disconnect errors)
)

SessionLocal = sessionmaker(autocommit=False, autoflush=False, bind=engine)
```

---

### 2. Breaking Down the Settings

#### A. `pool_size=5` and `max_overflow=10` (Total Max = 15)
* **Rule of thumb:** Your total maximum connections (`pool_size + max_overflow`) multiplied by the **number of microservice replicas** must not exceed PostgreSQL's `max_connections` limit (default is usually 100).
* If you run **4 replicas** of your FastAPI service, and each has a max pool of 15, that’s 60 connections—well within the default 100 limit, leaving room for migrations, admin tools, and Redis-backed workers (like Celery).
* **Why keep it small?** PostgreSQL performance degrades if it has to manage hundreds of idle connection threads. FastAPI's async nature means a small pool of connections can handle a *very* high throughput of concurrent requests, provided the queries themselves are fast.

#### B. `pool_pre_ping=True` (The Lifesaver)
* **What it does:** Sends a lightweight `SELECT 1` to the database before handing out a connection from the pool. If the database dropped the connection (due to a network blip, firewall timeout, or DB restart), SQLAlchemy automatically discards it and opens a fresh one.
* **Why you need it:** Without this, microservices frequently throw `OperationalError: server closed the connection unexpectedly` after periods of low traffic.

#### C. `pool_recycle=1800`
* **What it does:** Forces a connection to be retired and replaced after 30 minutes (1800 seconds), even if it's healthy.
* **Why you need it:** Cloud databases (like AWS RDS, Google Cloud SQL, or Heroku Postgres) often kill idle TCP connections after 30 to 60 minutes. Recycling prevents your app from trying to use a silently dead socket.

#### D. `pool_timeout=30`
* **What it does:** If all 15 connections are currently busy handling requests, a new incoming request will wait up to 30 seconds for a connection to be freed up. If none free up, it raises a timeout error.
* **Tuning:** Keep this reasonable (10–30s). If you hit this often, your database is either bottlenecked, or your `pool_size` needs a slight bump.

---

### 3. What if you are using Async SQLAlchemy (`asyncpg`)?

If you are using FastAPI's `async def` paths extensively and want non-blocking database I/O, you should use the **`asyncpg`** driver.

The configuration looks slightly different because `create_async_engine` manages its pool a bit differently:

```python
from sqlalchemy.ext.asyncio import create_async_engine

DATABASE_URL = "postgresql+asyncpg://user:password@localhost:5432/dbname"

engine = create_async_engine(
    DATABASE_URL,
    pool_size=10,  # Async can handle more concurrency per connection, but keep it bounded
    max_overflow=20,
    pool_timeout=30,
    pool_recycle=1800,
    pool_pre_ping=True,
)
```
*Note: `asyncpg` is significantly faster than `psycopg2`, but requires all your database calls to be `await`ed.*

---

### Summary Checklist for Deployment

1. **Check Postgres `max_connections`:** Run `SHOW max_connections;` in your DB. Ensure `(pool_size + max_overflow) * num_instances` is safely below that number.
2. **Always enable `pool_pre_ping=True`** in production environments.
3. **Use connection pooling at the app level, not a separate pouncer** (like PgBouncer) *unless* you are scaling to dozens of microservice replicas. For a standard microservice setup, SQLAlchemy's built-in pool is usually more than enough.

~/Projects/geni-ai-skd-intro master* 12s
geni-ai-skd-intro ❯
~/Projects/geni-ai-skd-intro master* 12s
geni-ai-skd-intro ❯ uv run practiced_examples/03_grounding_and_tools.py
Turn 1: That sounds like a robust, industry-standard technology stack. FastAPI gives you high performance and automatic OpenAPI docs, PostgreSQL provides reli ...

Turn 2: Configuring SQLAlchemy connection pooling for a **FastAPI + PostgreSQL** stack requires careful tuning. Because FastAPI handles requests asynchronously (using `asyncio`), but standard SQLAlchemy `Session` is synchronous, the way you manage connections is critical to avoiding bottlenecks and connection leaks.

Here are the recommended SQLAlchemy connection pooling settings, along with the reasoning behind them.

---

### 1. The Recommended Configuration

If you are using **`create_async_engine`** (recommended for async FastAPI), your setup should look something like this:

```python
from sqlalchemy.ext.asyncio import create_async_engine

DATABASE_URL = "postgresql+asyncpg://user:password@localhost/dbname"

engine = create_async_engine(
    DATABASE_URL,
    # --- Pooling Settings ---
    pool_size=20,  # Number of connections to keep open persistently
    max_overflow=10,  # Extra connections allowed if pool_size is exceeded
    pool_timeout=30,  # Seconds to wait for a connection before raising an error
    pool_recycle=1800,  # Recycle connections after 30 minutes (prevents stale connections)
    pool_pre_ping=True,  # Check connection liveness before using it (crucial for Postgres)
)
```

---

### 2. Breaking Down the Settings

#### A. `pool_size` (Default: 5)
* **What it does:** The number of connections to keep open in the pool during normal operation.
* **What to set it to:** Start between **10 to 20**.
* **Why:** The default of `5` is too low for most production web apps. However, don't set it to 100 either; PostgreSQL has a hard limit on connections (`max_connections`, usually 100 by default), and you need to leave room for migrations (Alembic), background workers (Celery/ARQ), and admin tools.

#### B. `max_overflow` (Default: 10)
* **What it does:** The number of *temporary* connections SQLAlchemy can create beyond `pool_size` under heavy load. Once released, these extra connections are closed.
* **What to set it to:** **10 to 20**.
* **Why:** This acts as a buffer for traffic spikes. If `pool_size` is 20 and `max_overflow` is 10, your app can handle up to 30 concurrent database operations at once.

#### C. `pool_pre_ping=True` (Default: False)
* **What it does:** Tests the database connection (`SELECT 1`) before handing it to your FastAPI endpoint.
* **What to set it to:** **`True`** (Always!)
* **Why:** Cloud providers (like AWS RDS, Heroku, or Supabase) and firewalls often drop idle TCP connections. Without `pool_pre_ping`, your app will occasionally crash with `OperationalError: SSL EOF detected` or `Server closed the connection unexpectedly` when trying to use a dead connection from the pool.

#### D. `pool_recycle` (Default: -1 / Never)
* **What it does:** The number of seconds a connection can live before it is automatically closed and replaced.
* **What to set it to:** **`1800` (30 minutes)** or **`3600` (1 hour)**.
* **Why:** Prevents issues with database-side timeouts (like `max_idle_transaction_timeout` or network NAT timeouts) dropping connections silently.

#### E. `pool_timeout` (Default: 30)
* **What it does:** How many seconds an application thread will wait for a connection to become available in the pool before throwing a `TimeoutError`.
* **What to set it to:** **`30`**.
* **Why:** If your pool is completely exhausted, it's better to fail fast after 30 seconds rather than hang indefinitely, allowing your web server to return a `504 Gateway Timeout` instead of freezing up.

---

### 3. The FastAPI Gotcha: Async vs. Sync Sessions

Because FastAPI is async, you must ensure your dependency injection correctly manages the lifecycle of the session so connections are **always returned to the pool**.

Here is the standard pattern for FastAPI with `async_sessionmaker`:

```python
from sqlalchemy.ext.asyncio import async_sessionmaker, async_session

# ... define engine with settings above ...

AsyncSessionLocal = async_sessionmaker(
    engine, expire_on_commit=False, class_=AsyncSession
)


async def get_db():
    async with AsyncSessionLocal() as session:
        try:
            yield session
            await session.commit()
        except Exception:
            await session.rollback()
            raise
        finally:
            await session.close()  # Crucial: Returns connection to pool
```

### Summary Checklist for Production:
1. Set `pool_size` based on your expected concurrent traffic and Postgres `max_connections`.
2. **Always** enable `pool_pre_ping=True`.
3. Set a `pool_recycle` interval (e.g., 1800s).
4. Use `async with AsyncSessionLocal()` in FastAPI dependencies to guarantee connection cleanup.
"""
#endregion

#region client_side_chat_demo()
"""client_side_chat_demo()
Chat Answer: You would have **22** working GPUs remaining.

(24 total GPUs - 2 failed GPUs = 22 working GPUs)
[user]: I have 3 servers, each with 8 GPUs....
[model]: That sounds like a very capable setup! With a total of 24 GPUs, you have a powerful cluster for machine learning, high-performance computing (HPC), rendering, o...
[user]: If 2 GPUs fail in total, how many working GPUs remain?...
[model]: You would have **22** working GPUs remaining.

(24 total GPUs - 2 failed GPUs = 22 working GPUs)...
"""
#endregion


solve_complex_architecture("A father and his son are in a car accident. The father dies. The son is rushed to the ER. The attending surgeon looks at the boy and says, 'I can't operate on this boy. He's my son!' How can this be?")