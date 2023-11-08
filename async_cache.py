import asyncio
import pickle
import os
import random
from threading import RLock
from filelock import FileLock, Timeout
import asyncio
import functools
import time
from multiprocessing import Manager
from concurrent.futures import ProcessPoolExecutor
import multiprocessing
import asyncio
import atexit
from concurrent.futures import ProcessPoolExecutor

ASYNC_DEFAULT_CACHE_FILENAME = "async_cache"
ASYNC_DEFAULT_CACHE_DIR = "./"

class CachedDict(dict):
    def __init__(self, local_cache_dir=ASYNC_DEFAULT_CACHE_DIR, 
                       local_cache_keyname=ASYNC_DEFAULT_CACHE_FILENAME,
                       clear_if_exists=False, 
                       *args, **kwargs):

        super().__init__()
        self.update(*args, **kwargs)
        if local_cache_dir:
            self.file_path = self._file_path_from_key_name(local_cache_keyname, local_cache_dir)
            self.lock_path = f'{self.file_path}.lock'
            self.old_path = f'{self.file_path}.old'
            self.file_lock = FileLock(self.lock_path, timeout=-1)               
            if not clear_if_exists:
                self.update(self.read_cache())
        else:
            self.file_path = None
    
    def read_cache(self):
        try:
            with self.file_lock:
                with open(self.file_path, 'rb') as f:
                    return pickle.load(f)#_do_pickle_file_load(f)
        except (EOFError, pickle.UnpicklingError, FileNotFoundError):
            # This almost certainly means the pickled file did not finish fully writing
            # likely due to a wonky exit. Try and find an old version if exists.
            try:
                with self.file_lock:
                    with open(self.old_path, 'rb') as f:
                        return pickle.load(f)#_do_pickle_file_load(f)
            except (EOFError, FileNotFoundError):
                # raise FileNotFoundError
                pass
        return {}

    def save_cache(self):
        if not self.file_path:
            return
        try:
            with self.file_lock:
                try:
                    # We keep the old file in case we get a write error due to bad exit
                    os.remove(self.old_path)
                except FileNotFoundError:
                    pass

                try:
                    os.rename(self.file_path, self.old_path)
                except FileNotFoundError:
                    pass

                with open(self.file_path, 'wb') as f:
                    pickle.dump(dict(self), f, protocol=5)
        except Timeout:
            os.remove(self.lock_path)  # Assume because of old bad shutdown
            self.save_cache()
        
    def _file_path_from_key_name(self, keyname, path):
        if not keyname.endswith('.acache'):
            keyname = f'{keyname}.acache'
        os.makedirs(path, exist_ok=True)
        return os.path.join(path, keyname)





def run_pricefeed(queue):
    i=0
    queue.put_nowait(f'test-{i}')
    #    i += 1
    #    time.sleep(.25)


class CachedQueue(object):
    def __init__(self, local_cache_dir=ASYNC_DEFAULT_CACHE_DIR, 
                       local_cache_keyname=ASYNC_DEFAULT_CACHE_FILENAME,
                       clear_if_exists=False, 
                       *args, **kwargs): 
        #self.manager = Manager() 
        self.queue = asyncio.Queue()
        #self.queue = None
        #self.loop = asyncio.get_event_loop()
        
    async def queue_subprocess(self):
        print('in process')
        while True:
            if not self.queue.empty():
                msg = self.queue.get(block=False)
                print(msg)
                self.queue.task_done()
            #await asyncio.sleep(1)  # do some stuff

    def start(self):
        asyncio.run(e._run())
    
    async def _run(self):
        #with ProcessPoolExecutor() as pool:
            #with Manager() as manager:
        await asyncio.create_task(self.queue_subprocess())
        #await self.queue.join()
                #await queue.join()
                #await self.run_forever()
                #await asyncio.get_running_loop().run_in_executor(pool, functools.partial(run_pricefeed, queue))


#async def main():
#    global g
#    g = Executor()
#    await g.run()

if __name__ == '__main__':
    e = CachedQueue()
    #asyncio.run(e.run())
    e.start()
    print('ok')
    time.sleep(2)
    run_pricefeed(e.queue)
    time.sleep(5)
    







#c = CachedDict(clear_if_exists=False)
#print(c)
#c[f'test{random.randint(0,100)}'] = 't'
#c.save_cache()
#print(c)


