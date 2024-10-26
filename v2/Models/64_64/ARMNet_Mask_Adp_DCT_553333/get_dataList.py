from dataset import GetDataList

if __name__=='__main__':
    getDataList=GetDataList()
    getDataList.get_dataList(data_paths=['F:/MRI Denoising v2/data/train/240_240/'], save_path='./dataDiv_list/', save_name='train_list.txt')
    getDataList.get_dataList(data_paths=['F:/MRI Denoising v2/data/validation/240_240/'], save_path='./dataDiv_list/', save_name='eval_list.txt')
