import glob
import os
import sys

from graduation.tools.pefeatures import PEFeatureExtractor

module_path = os.path.dirname(os.path.abspath(sys.modules[__name__].__file__))

SAMPLE_PATH = os.path.join(module_path, 'samples')

# for local model
from sklearn.externals import joblib

feature_extractor = PEFeatureExtractor()
local_model = joblib.load(os.path.join(module_path, 'gradient_boosting.pkl'))
local_model_threshold = 0.90


class ClassificationFailure(Exception):
    pass


class FileRetrievalFailure(Exception):
    pass


# 获取文件二进制数据
def fetch_file(sha256):
    location = os.path.join(SAMPLE_PATH, sha256)
    try:
        with open(location, 'rb') as infile:
            bytez = infile.read()
    except IOError:
        raise FileRetrievalFailure(
            "Unable to read sha256 from {}".format(location))

    return bytez


def delete_file(sha256):
    location = os.path.join(SAMPLE_PATH, sha256)
    os.remove(location)


# 在samples目录中读取样本，放入list返回
def get_available_sha256():
    sha256list = []
    for fp in glob.glob(os.path.join(SAMPLE_PATH, '*')):
        fn = os.path.split(fp)[-1]
        # result = re.match(r'^[0-9a-fA-F]{64}$', fn) # require filenames to be sha256
        # if result:
        #     sha256list.append(result.group(0))
        sha256list.append(fn)
    assert len(sha256list) > 0, "no files found in {} with sha256 names".format(SAMPLE_PATH)
    return sha256list


# 从samples目录中找到某个文件名的样本路径
def get_sample_real_path(filename):
    return os.path.join(SAMPLE_PATH, filename)


# 获取分类器评分
def get_score_local(bytez):
    # extract features
    features = feature_extractor.extract(bytez)

    # query the model
    score = local_model.predict_proba(features.reshape(1, -1))[
        0, -1]  # predict on single sample, get the malicious score
    return score


# 获取分类器label
def get_label_local(bytez):
    # mimic black box by thresholding here
    score = get_score_local(bytez)
    label = float(get_score_local(bytez) >= local_model_threshold)
    return label
