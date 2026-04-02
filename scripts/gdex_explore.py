from os import login_tty
from gdex_api_client import gdex_client as rc
from pprint import pprint
import datetime
import numpy as np

dataset_no = "d633000"
result = rc.get_param_summary(dataset_no)
pprint(result)
result = rc.get_metadata(dataset_no)
pprint(result)
result = rc.get_control_file_template(dataset_no)
pprint(result)
template_dict = rc.read_control_file(result["data"]["template"])
pprint(template_dict)

start_date = datetime.datetime(2024, 1, 1, 0, 0)
end_date = datetime.datetime(2024, 12, 31, 23, 0)
params_list = ["VAR_U10", "VAR_U100", "VAR_V10", "VAR_V100", "FSR"]
lon = np.array([52, 53])
lat = np.array([13, 14])

time_format_string = "%Y%m%d%H%M"


template_dict = {}
template_dict["compression"] = "gzip"
template_dict["dataset"] = dataset_no
template_dict["date"] = (
    start_date.strftime(time_format_string)
    + "/to/"
    + end_date.strftime(time_format_string)
)
template_dict["param"] = ("/").join(params_list)
template_dict["elon"] = str(lon.max())
template_dict["wlon"] = str(lon.min())
template_dict["nlat"] = str(lat.max())
template_dict["slat"] = str(lat.min())

pprint(template_dict)

response = rc.submit_json(template_dict)
request_id = response["data"]["request_id"]
filelist = rc.get_filelist(request_id)
# rc.download(request_id)
#
