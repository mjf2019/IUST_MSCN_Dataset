"""Check timing boundaries, GPU completion and accounting without sleeps."""
import sys
from pathlib import Path
import unittest
from unittest.mock import patch

sys.path.insert(0,str(Path(__file__).resolve().parents[1]))
from baselines.stream_timing import StreamTimer


class StreamTimingTests(unittest.TestCase):
    def test_cache_is_charged_once_and_observer_time_is_excluded(self):
        with patch('baselines.stream_timing.time.perf_counter',side_effect=[0,1,100,101,103,104,105,106,110,111]):
            timer=StreamTimer('cpu')
            with timer.stage('cache'):pass
            # Unmeasured initial observer occupies the time gap 1..100.
            timer.start()
            with timer.stage('prediction'):pass
            with timer.stage('selection'):pass
            with timer.stage('update'):pass
            timer.stop()
        m=timer.metrics(4,2)
        self.assertEqual(m['online_seconds'],12)
        self.assertEqual(m['prediction_seconds'],3)
        self.assertEqual(m['cache_seconds'],1)
        self.assertEqual(m['other_seconds'],4)
        self.assertEqual(m['online_ms_per_sample'],3000)
        self.assertEqual(m['update_ms_per_label'],2000)
        self.assertEqual(timer.metrics(4,0)['update_ms_per_label'],0)

    def test_gpu_completion_is_waited_for_at_boundaries(self):
        with patch('baselines.stream_timing.torch.cuda.synchronize') as sync:
            timer=StreamTimer('cuda:0');timer.start()
            with timer.stage('prediction'):pass
            timer.stop()
        self.assertEqual(sync.call_count,4)

    def test_offline_training_does_not_collect_runtime(self):
        with patch('baselines.stream_timing.time.perf_counter') as clock:
            timer=StreamTimer('cpu',enabled=False);timer.start()
            with timer.stage('update'):pass
            timer.stop()
        clock.assert_not_called()
        self.assertEqual(timer.metrics(10,2),{})


if __name__=='__main__':unittest.main()
