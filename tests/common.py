# """
# Common utils for testing.

# https://github.com/kornia/kornia/blob/0b94940f31114060b80fa6d35384c444ba81b540/test/common.py
# """

# from typing import Dict

# import torch
# import pytest


# def get_test_devices() -> Dict[str, torch.device]:
#     r"""Creates a dictionary of test devices available.

#     Return:
#         dict(str, torch.device): Dictionary of available device ids.
#     """
#     devices: Dict[str, torch.device] = {}
#     devices["cpu"] = torch.device("cpu")
#     if torch.cuda.is_available():
#         devices["cuda"] = torch.device("cuda:0")
#     return devices


# # Setup test devices.
# TEST_DEVICES: Dict[str, torch.device] = get_test_devices()

# @pytest.fixture()
# def device(request) -> torch.device:
#     _device_type: str = request.config.getoption('--typetest')
#     return TEST_DEVICES[_device_type]
