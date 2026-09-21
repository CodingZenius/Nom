import unittest
from nomad import hardware


def fake(avail, gpu=None, cores=4):
    hw = {"arch": "amd64", "cpu_model": "x", "cpu_flags": [], "cores": cores, "eff_cores": cores,
          "ram_total_gb": avail + 1, "ram_avail_gb": avail, "gpu": gpu}
    return hw


class TestHardware(unittest.TestCase):
    def test_detect_runs(self):
        hw = hardware.detect()
        self.assertGreaterEqual(hw["eff_cores"], 1)
        self.assertEqual(len(hw["fingerprint"]), 12)

    def test_fingerprint_changes_with_cpu(self):
        a = fake(8)
        b = dict(a, cpu_model="other")
        self.assertNotEqual(hardware.fingerprint(a), hardware.fingerprint(b))
        c = dict(a, ram_avail_gb=2)  # free RAM fluctuates: must NOT change the fingerprint
        self.assertEqual(hardware.fingerprint(a), hardware.fingerprint(c))

    def test_ctx_scaling(self):
        big = hardware.recommend(fake(16), "qwen2.5:3b", 32768)
        self.assertEqual(big["num_ctx"], 8192)                 # CPU cap
        gpu = hardware.recommend(fake(16, gpu="T4"), "qwen2.5:3b", 32768)
        self.assertEqual(gpu["num_ctx"], 32768)
        tight = hardware.recommend(fake(3.2), "qwen2.5:3b", 32768)
        self.assertLessEqual(tight["num_ctx"], 4096)
        self.assertTrue(tight["notes"] or tight["num_ctx"] >= 2048)
        capped = hardware.recommend(fake(16), "qwen2.5:3b", 4096)
        self.assertEqual(capped["num_ctx"], 4096)
        self.assertEqual(hardware.recommend(fake(16, cores=6), "qwen2.5:3b")["num_thread"], 6)


if __name__ == "__main__":
    unittest.main()
