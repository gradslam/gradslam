import logging
import os
import pytest
import sys
import tempfile

from gradslam.config import CfgNode as CN


class SubCN(CN):
    pass


def get_cfg(cls=CN):
    cfg = cls()

    cfg.NUM_GPUS = 1

    cfg.TRAIN = cls()
    cfg.TRAIN.HYPERPARAM_1 = 0.1
    cfg.TRAIN.SCALES = (2, 4, 8, 16)

    cfg.MODEL = cls()
    cfg.MODEL.TYPE = "a_foo_model"

    cfg.STR = cls()
    cfg.STR.KEY1 = 1
    cfg.STR.KEY2 = 2
    cfg.STR.FOO = cls()
    cfg.STR.FOO.KEY1 = 1
    cfg.STR.FOO.KEY2 = 2
    cfg.STR.FOO.BAR = cls()
    cfg.STR.FOO.BAR.KEY1 = 1
    cfg.STR.FOO.BAR.KEY2 = 2

    cfg.register_deprecated_key("FINAL_MSG")
    cfg.register_deprecated_key("MODEL.DILATION")

    cfg.register_renamed_key(
        "EXAMPLE.OLD.KEY", "EXAMPLE.NEW.KEY", message="Please update your config file.",
    )

    cfg.KWARGS = cls(new_allowed=True)
    cfg.KWARGS.z = 0
    cfg.KWARGS.Y = cls()
    cfg.KWARGS.Y.X = 1

    return cfg


class TestCfgNode:
    def test_immutability(self):
        # Top level immutable
        a = CN()
        a.foo = 0
        a.freeze()
        with pytest.raises(AttributeError):
            a.foo = 1
            a.bar = 1
        assert a.is_frozen()
        assert a.foo == 0
        a.defrost()
        assert not a.is_frozen()
        a.foo = 1
        assert a.foo == 1

        # Recursively immutable
        a.level1 = CN()
        a.level1.foo = 0
        a.level1.level2 = CN()
        a.level1.level2.foo = 0
        a.freeze()
        assert a.is_frozen()
        with pytest.raises(AttributeError):
            a.level1.level2.foo = 1
            a.level1.bar = 1
        assert a.level1.level2.foo == 0


class TestCfg:
    def test_copy_cfg(self):
        cfg = get_cfg()
        cfg2 = cfg.clone()
        s = cfg.MODEL.TYPE
        cfg2.MODEL.TYPE = "a_foo_model"
        assert cfg.MODEL.TYPE == s

    def test_merge_cfg_from_cfg(self):
        # Test: merge from clone
        cfg = get_cfg()
        s = "dummy0"
        cfg2 = cfg.clone()
        cfg2.MODEL.TYPE = s
        cfg.merge_from_other_cfg(cfg2)
        assert cfg.MODEL.TYPE == s

        # Test: merge from yaml
        s = "dummy1"
        cfg2 = CN.load_cfg(cfg.dump())
        cfg2.MODEL.TYPE = s
        cfg.merge_from_other_cfg(cfg2)

        # Test: merge with a valid key
        s = "dummy2"
        cfg2 = CN()
        cfg2.MODEL = CN()
        cfg2.MODEL.TYPE = s
        cfg.merge_from_other_cfg(cfg2)
        assert cfg.MODEL.TYPE == s

        # Test: merge with an invalid key
        s = "dummy3"
        cfg2 = CN()
        cfg2.FOO = CN()
        cfg2.FOO.BAR = s
        with pytest.raises(KeyError):
            cfg.merge_from_other_cfg(cfg2)

        # Test: merge with converted type
        cfg2 = CN()
        cfg2.TRAIN = CN()
        cfg2.TRAIN.SCALES = [1]
        cfg.merge_from_other_cfg(cfg2)
        assert type(cfg.TRAIN.SCALES) is tuple
        assert cfg.TRAIN.SCALES[0] == 1

        # Test: merge with invalid type
        cfg2 = CN()
        cfg2.TRAIN = CN()
        cfg2.TRAIN.SCALES = 1
        with pytest.raises(ValueError):
            cfg.merge_from_other_cfg(cfg2)

    def test_merge_cfg_from_file(self):
        with tempfile.NamedTemporaryFile(mode="wt") as f:
            cfg = get_cfg()
            f.write(cfg.dump())
            f.flush()
            s = cfg.MODEL.TYPE
            cfg.MODEL.TYPE = "dummy"
            assert cfg.MODEL.TYPE != s
            cfg.merge_from_file(f.name)
            assert cfg.MODEL.TYPE == s

    def test_merge_cfg_from_list(self):
        cfg = get_cfg()
        opts = ["TRAIN.SCALES", "(100, )", "MODEL.TYPE", "foobar", "NUM_GPUS", 2]
        assert len(cfg.TRAIN.SCALES) > 0
        assert cfg.TRAIN.SCALES[0] != 100
        assert cfg.MODEL.TYPE != "foobar"
        assert cfg.NUM_GPUS != 2
        cfg.merge_from_list(opts)
        assert type(cfg.TRAIN.SCALES) is tuple
        assert len(cfg.TRAIN.SCALES) == 1
        assert cfg.TRAIN.SCALES[0] == 100
        assert cfg.MODEL.TYPE == "foobar"
        assert cfg.NUM_GPUS == 2

    def test_deprecated_key_from_list(self):
        # You should see logger messages like:
        #  "Deprecated config key (ignoring): MODEL.DILATION"
        cfg = get_cfg()
        opts = ["FINAL_MSG", "foobar", "MODEL.DILATION", 2]
        with pytest.raises(AttributeError):
            _ = cfg.FINAL_MSG  # noqa
        with pytest.raises(AttributeError):
            _ = cfg.MODEL.DILATION  # noqa
        cfg.merge_from_list(opts)
        with pytest.raises(AttributeError):
            _ = cfg.FINAL_MSG  # noqa
        with pytest.raises(AttributeError):
            _ = cfg.MODEL.DILATION  # noqa

    def test_nonexistent_key_from_list(self):
        cfg = get_cfg()
        opts = ["MODEL.DOES_NOT_EXIST", "IGNORE"]
        with pytest.raises(AssertionError):
            cfg.merge_from_list(opts)

    def test_load_cfg_invalid_type(self):
        # FOO.BAR.QUUX will have type None, which is not allowed
        cfg_string = "FOO:\n BAR:\n QUUX:"
        with pytest.raises(AssertionError):
            CN.load_cfg(cfg_string)

    def test_deprecated_key_from_file(self):
        # You should see logger messages like:
        #  "Deprecated config key (ignoring): MODEL.DILATION"
        cfg = get_cfg()
        with tempfile.NamedTemporaryFile("wt") as f:
            cfg2 = cfg.clone()
            cfg2.MODEL.DILATION = 2
            f.write(cfg2.dump())
            f.flush()
            with pytest.raises(AttributeError):
                _ = cfg.MODEL.DILATION  # noqa

    def test_renamed_key_from_list(self):
        cfg = get_cfg()
        opts = ["EXAMPLE.OLD.KEY", "foobar"]
        with pytest.raises(AttributeError):
            _ = cfg.EXAMPLE.OLD.KEY  # noqa
        with pytest.raises(KeyError):
            cfg.merge_from_list(opts)

    def test_renamed_key_from_file(self):
        cfg = get_cfg()
        with tempfile.NamedTemporaryFile("wt") as f:
            cfg2 = cfg.clone()
            cfg2.EXAMPLE = CN()
            cfg2.EXAMPLE.RENAMED = CN()
            cfg2.EXAMPLE.RENAMED.KEY = "foobar"
            f.write(cfg2.dump())
            f.flush()
            with pytest.raises(AttributeError):
                _ = cfg.EXAMPLE.RENAMED.KEY  # noqa
            with pytest.raises(KeyError):
                cfg.merge_from_file(f.name)

    def test_load_cfg_from_file(self):
        cfg = get_cfg()
        with tempfile.NamedTemporaryFile("wt") as f:
            f.write(cfg.dump())
            f.flush()
            with open(f.name, "rt") as f_read:
                CN.load_cfg(f_read)

    def test_load_from_python_file(self):
        # Case 1: exports CfgNode
        cfg = get_cfg()
        cfg.merge_from_file("tests/data/samplecfg.py")
        assert cfg.TRAIN.HYPERPARAM_1 == 0.9
        # Case 2: exports dict
        cfg = get_cfg()
        cfg.merge_from_file("tests/data/samplecfg_dict.py")
        assert cfg.TRAIN.HYPERPARAM_1 == 0.9

    def test_invalid_type(self):
        cfg = get_cfg()
        with pytest.raises(AssertionError):
            cfg.INVALID_KEY_TYPE = object()

    def test__str__(self):
        expected_str = """
KWARGS:
  Y:
    X: 1
  z: 0
MODEL:
  TYPE: a_foo_model
NUM_GPUS: 1
STR:
  FOO:
    BAR:
      KEY1: 1
      KEY2: 2
    KEY1: 1
    KEY2: 2
  KEY1: 1
  KEY2: 2
TRAIN:
  HYPERPARAM_1: 0.1
  SCALES: (2, 4, 8, 16)
""".strip()
        cfg = get_cfg()
        assert str(cfg) == expected_str

    def test_new_allowed(self):
        cfg = get_cfg()
        cfg.merge_from_file("tests/data/cfg_new_allowed.yaml")
        assert cfg.KWARGS.a == 1
        assert cfg.KWARGS.B.c == 2
        assert cfg.KWARGS.B.D.e == "3"

    def test_new_allowed_bad(self):
        cfg = get_cfg()
        with pytest.raises(KeyError):
            cfg.merge_from_file("tests/data/cfg_new_allowed_bad.yaml")


if __name__ == "__main__":
    logging.basicConfig()
    logger = logging.getLogger("config")
    logger.setLevel(logging.DEBUG)
    unittest.main()
