"""Official Unicorn-v1 RWTT Attribute reference operating points.

This mapping follows lossy_attribute/test.py's object configuration after
removing the overlapping 8k256/lambda=8192 boundary point.  Do not infer a
checkpoint from lambda or invent additional reference combinations.
"""


OFFICIAL_RWTT_REFERENCE_POINTS = (
    ("R01", "32k8k", 32768),
    ("R02", "32k8k", 16384),
    ("R03", "32k8k", 8192),
    ("R04", "8k256", 4096),
    ("R05", "8k256", 2048),
    ("R06", "2k128", 1024),
    ("R07", "2k128", 512),
    ("R08", "2k128", 256),
    ("R09", "2k128", 128),
)


def reference_checkpoint(profile):
    return (
        "unicorn_released/Unicorn-v1-attribute-test-only-weights/ckpts/"
        "lossy_attribute/rwtt/{}/epoch_last.pth".format(profile)
    )


def reference_point(rate_id):
    matches = [item for item in OFFICIAL_RWTT_REFERENCE_POINTS
               if item[0] == rate_id]
    if not matches:
        raise ValueError("Unknown official RWTT rate ID: " + rate_id)
    return matches[0]
