from dataset import PreProcessing

if __name__=='__main__':
    preProcessing=PreProcessing()
    preProcessing.process(save_path='./dataDiv_list/', eval_ratio=0.1, test_ratio=0.1)
