import os.path
import sys 

from pydrive2.auth import GoogleAuth
from pydrive2.drive import GoogleDrive
import rasterio
import numpy as np

import argparse
import asyncio



import logging

logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

gauth = GoogleAuth()
print("auth")
gauth.LocalWebserverAuth()

drive = GoogleDrive(gauth)


current_script_dir = os.path.dirname(os.path.abspath(__file__))  # src/gee_export/
parent_dir = os.path.dirname(current_script_dir)  # src/
project_root = os.path.dirname(parent_dir)  # BDA_integration/

# Add the src directory to Python path so we can import worker_queue directly
sys.path.insert(0, parent_dir)

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
    
    async def wait_all(self):
        while True:
            async with self._lock:
                if not self.running and not self.waiting:
                    break
                current_running = list(self.running)
            
            if current_running:
                await asyncio.wait(current_running, return_when=asyncio.FIRST_COMPLETED)
            else:
                await asyncio.sleep(0.01)
    
    def stats(self):
        """Get queue statistics"""
        return {
            'running': len(self.running),
            'waiting': len(self.waiting),
            'completed': len(self.completed),
            'failed': len(self.failed)
        }



async def download_file_content(file, 
                          destination_folder,
                          prefix="X",
                          tempdir="tempdir",
                          saveas=".npy" #.npy or .tif
                          ):
  try:
    await asyncio.sleep(0.1)

    filename = file["title"]
    id = "".join([c for c in filename if c.isdigit()])

    final_path = os.path.join(destination_folder, f"{prefix}_{id}{saveas}")
    if os.path.exists(final_path):
      return
    if filename.endswith(saveas):
      file.GetContentFile(final_path)
    elif filename.endswith(".tif") and saveas == ".npy":
      tempdest = os.path.join(tempdir, f"{prefix}_{id}.tif")
      file.GetContentFile(tempdest)
      with rasterio.open(tempdest) as infile:
        content = infile.read()
        np.save(final_path, content)
      os.remove(tempdest)
    elif filename.endswith(".npy") and saveas == ".tif":
      raise NotImplementedError("Conversation from npy to tif is not implemented yet")
    else:
        suffix = filename.split(".")[-1]
        raise NotImplementedError(f"Unsupported filetype:{suffix}")
    
  except Exception as e:
    print(f"Exception occurred: {e}")
    raise e
  
async def main(folder_id,
               destination_folder,
               prefix,
              limit,
              tempdir,
              names=None,
              saveas=".npy",
               ):
  wq = WorkerQueue(logger=logger,)

  file_list = drive.ListFile({'q': f"'{folder_id}' in parents and trashed=false"}).GetList()
  if limit > -1:
    file_list = file_list[:limit]
  length = len(file_list)

  for i in range(length):
    current_file = file_list[i]
    if names and current_file["title"] not in names:
      continue
    await wq.add_task(download_file_content(current_file, destination_folder, prefix, saveas=saveas, tempdir=tempdir))
                                           
    
    if i % 10 == 0:
        stats = wq.stats()
        logger.info(f"Progress: {i}/{length}, Queue stats: {stats}")
    await wq.wait_all()

if __name__ == "__main__":
  parser = argparse.ArgumentParser()

  parser.add_argument("destination_folders", nargs="+", help="The ultimate folder to save the downloaded information to")
  parser.add_argument("--folder_ids", nargs="+", default="1zZZN9gCRqUApYsXNsM4UsDZDLbEO7waK", help="The id of the google drive folder whose contents are to be downloaded")
  parser.add_argument("--prefixes", nargs="+", default="X", help="The prefix to prepend to the downloaded files(usually one of 'X' or 'y')")
  parser.add_argument("--limit", default=-1, help="The number of files to download. Defaults to downloading all files.")
  parser.add_argument('--names', nargs='*', default=None)
  parser.add_argument('--saveas',  default=".npy")
  parser.add_argument('--tempdir')
  
  args = parser.parse_args()

  if args.saveas not in [".npy", ".tif"]:
    raise Exception(f"Only npy and tif are supported, you passed {args.saveas}")
  
  runs = zip(args.folder_ids, args.destination_folders, args.prefixes)
  for folder_id, destination_folder, prefix in runs:
    asyncio.run(main(
      folder_id = folder_id,
      destination_folder = destination_folder,
      prefix = prefix,
      limit=int(args.limit),
      names=args.names,
      saveas=args.saveas,
      tempdir=args.tempdir
    ))
  


