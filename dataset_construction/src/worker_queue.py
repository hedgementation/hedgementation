from collections import deque
import random
import asyncio
"""
Worker queue to help with the asynchronous rasterization/export pipeline in rasterization.py
"""

class WorkerQueue:
    def __init__(self, logger,max_concurrent=5):
        self.max_concurrent = max_concurrent
        self.running = set()
        self.waiting = deque()
        self.completed = []
        self.failed = []
        self.logger = logger
        self._lock = asyncio.Lock()

    async def add_task(self, coro):
        async with self._lock:
            if len(self.running) >= self.max_concurrent:
                self.waiting.append(coro)
            else:
                await self._start_task(coro)
    
    async def _start_task(self, coro):
        task = asyncio.create_task(self._execute_task(coro))
        self.running.add(task)
        task.add_done_callback(self._task_done)

    def _task_done(self, task):
        asyncio.create_task(self._handle_task_completion(task))

    async def _handle_task_completion(self, completed_task):
        async with self._lock:
            self.running.discard(completed_task)
            
            if self.waiting and len(self.running) < self.max_concurrent:
                next_coro = self.waiting.popleft()
                await self._start_task(next_coro)

    async def _execute_task(self, coro):
        try:
            result = await coro
            self.completed.append(result)
            return result
        except Exception as e:
            self.logger.error(f"Task failed: {e}")
            self.failed.append(e)
            raise 
    
    async def wait_all(self, stats_interval: float = 5.0):
        last_logged = asyncio.get_event_loop().time()
        last_completed = 0

        while True:
            async with self._lock:
                if not self.running and not self.waiting:
                    break
                current_running = list(self.running)

            now = asyncio.get_event_loop().time()
            completed_now = len(self.completed)

            if (now - last_logged >= stats_interval) or (completed_now != last_completed):
                s = self.stats()
                self.logger.info(
                    f"Queue | running={s['running']} waiting={s['waiting']} "
                    f"completed={s['completed']} failed={s['failed']}"
                )
                last_logged = now
                last_completed = completed_now

            if current_running:
                await asyncio.wait(current_running, return_when=asyncio.FIRST_COMPLETED)
            else:
                await asyncio.sleep(0.01)

        s = self.stats()
        self.logger.info(
            f"Queue done | completed={s['completed']} failed={s['failed']}"
        )
    
    def stats(self):
        """Get queue statistics"""
        return {
            'running': len(self.running),
            'waiting': len(self.waiting),
            'completed': len(self.completed),
            'failed': len(self.failed)
        }
