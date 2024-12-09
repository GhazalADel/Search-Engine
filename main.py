import time

import models.dataset
from parsivar import Normalizer
import pickle

def main():
    dataset = models.dataset.Dataset(dataset_path="initial_data/IR_data_news_12k.json")
    dataset.read_dataset()
    preprocessed_docs = dataset.preprocess()
    # print(dataset.read_content_at_index(14))
    positional_index = dataset.positional_index()
    #print(len(dataset.get_dictionary()))
    # with open('tiny.pkl', 'rb') as f:
    #     dataset = pickle.load(f)  # deserialize using load()
    dataset.calculate_weights()

    #print(len(dataset.get_dictionary()))
    #print(list(sorted(dataset.tfff,key=lambda x : x[1],reverse=False))[:10])
    # with open('posindx.pkl', 'wb') as f:
    #     pickle.dump(positional_index, f)
    # f.close()
    # dataset = None
    # with open('tiny.pkl', 'rb') as f:
    #     dataset = pickle.load(f)  # deserialize using load()




    #print(positional_index['هندبال'].position_in_docs)
    # if positional_index.get('فارس') is not None:
    #     print(positional_index.get('فارس').total_frequency)  # tedad tekrar kalame dar kol collection - adad

    # print(positional_index['فوتبال'].position_in_docs) #map - har doc id ra be yek list map mikonad va list havi positiona haye kalame dar on doc
    # print(positional_index['فوتبال'].frequency_in_docs) #map - har doc id ra be yek adad map mikonad va value barabar tedad tekrar kalame dar on document
    # print(len(dataset.get_dictionary()))
    # print(dataset.get_dictionary())

    # print(positional_index['فوتبال'].weight_in_docs)
    # print(dataset.docs_norm)
    #print(dataset.data["3597"])
    dataset.norm_docs()
    # dataset.create_champion_list(300)
    # start = time.time()
    res = dataset.k_nearest_documents('مجمع عمومی حزب',3)
    # #print(res)
    for r in res:
        title = dataset.read_title_at_index(r)
        #url = dataset.read_url_at_index(r)
        content = dataset.read_content_at_index(r)
        print(title)
        #print(url)
        print(content)
        print('x------------x------------x------------x------------x------------x------------x------------x------------x')
    #
    # end = time.time()
    # print(end - start)
    # dataset.no_norm_docs()
    # res = dataset.k_nearest_documents('فوتبال', 3)
    # # print(res)
    # for r in res:
    #     title = dataset.read_title_at_index(r)
    #     url = dataset.read_url_at_index(r)
    #     content = dataset.read_content_at_index(r)
    #     print(title)
    #     print(url)
    #     print(content)
    #     print('x------------x------------x------------x------------x------------x------------x------------x------------x')



if __name__ == '__main__':
    main()