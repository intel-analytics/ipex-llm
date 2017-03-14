import tarfile
import base
import os
import sys

NEWS20_URL = 'http://qwone.com/~jason/20Newsgroups/20news-19997.tar.gz'  # noqa
GLOVE_URL = 'http://nlp.stanford.edu/data/glove.6B.zip'  # noqa

CLASS_NUM = 20


def download_news20(dest_dir):
    file_name = "20news-19997.tar.gz"
    file_abs_path = base.maybe_download(file_name, dest_dir, NEWS20_URL)
    tar = tarfile.open(file_abs_path, "r:gz")
    extracted_to = os.path.join(dest_dir, "20_newsgroups")
    if not os.path.exists(extracted_to):
        print("Extracting %s to %s" % (file_abs_path, extracted_to))
        tar.extractall(dest_dir)
        tar.close()
    return extracted_to


def download_glove_w2v(dest_dir):
    file_name = "glove.6B.zip"
    file_abs_path = base.maybe_download(file_name, dest_dir, GLOVE_URL)
    import zipfile
    zip_ref = zipfile.ZipFile(file_abs_path, 'r')
    extracted_to = os.path.join(dest_dir, "glove.6B")
    if not os.path.exists(extracted_to):
        print("Extracting %s to %s" % (file_abs_path, extracted_to))
        zip_ref.extractall(extracted_to)
        zip_ref.close()
    return extracted_to


# return [(text_content, label)]
def get_news20(source_dir="/tmp/news20/"):
    news_dir = download_news20(source_dir)
    texts = []  # list of text samples
    label_id = 0
    for name in sorted(os.listdir(news_dir)):
        path = os.path.join(news_dir, name)
        label_id += 1
        if os.path.isdir(path):
            for fname in sorted(os.listdir(path)):
                if fname.isdigit():
                    fpath = os.path.join(path, fname)
                    if sys.version_info < (3,):
                        f = open(fpath)
                    else:
                        f = open(fpath, encoding='latin-1')
                    content = f.read()
                    texts.append((content, label_id))
                    f.close()

    print('Found %s texts.' % len(texts))
    return texts


def get_glove_w2v(source_dir="/tmp/news20/", dim=100):
    w2v_dir = download_glove_w2v(source_dir)
    with open(os.path.join(w2v_dir, "glove.6B.%sd.txt" % dim)) as w2v_f:
        pre_w2v = {}
        for line in w2v_f.readlines():
            items = line.split(" ")
            pre_w2v[items[0]] = [float(i) for i in items[1:]]
        return pre_w2v


if __name__ == "__main__":
    get_news20("/tmp/news20/")
    get_glove_w2v("/tmp/news20/")
