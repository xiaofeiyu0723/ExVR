import itertools
import queue
import threading


ORT_PRIORITY_HAND = 0
ORT_PRIORITY_FACE = 10


class _OrtRunScheduler:
    def __init__(self):
        self._queue = queue.PriorityQueue()
        self._counter = itertools.count()
        self._worker = threading.Thread(
            target=self._worker_loop, daemon=True, name="OrtRunScheduler"
        )
        self._worker.start()

    def run(self, session, output_names, input_feed, priority):
        if "DmlExecutionProvider" not in session.get_providers():
            return session.run(output_names, input_feed)

        done = threading.Event()
        request = {
            "session": session,
            "output_names": output_names,
            "input_feed": input_feed,
            "done": done,
            "result": None,
            "error": None,
        }
        self._queue.put((priority, next(self._counter), request))
        done.wait()
        if request["error"] is not None:
            raise request["error"]
        return request["result"]

    def _worker_loop(self):
        while True:
            _, _, request = self._queue.get()
            try:
                request["result"] = request["session"].run(
                    request["output_names"], request["input_feed"]
                )
            except Exception as exc:
                request["error"] = exc
            finally:
                request["done"].set()
                self._queue.task_done()


_ORT_RUN_SCHEDULER = _OrtRunScheduler()


def run_ort(session, output_names, input_feed, priority=ORT_PRIORITY_FACE):
    return _ORT_RUN_SCHEDULER.run(session, output_names, input_feed, priority)
