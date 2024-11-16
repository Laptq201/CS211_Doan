# CEM-RL
Pytorch implementation of CEM-RL: https://arxiv.org/pdf/1810.01222.pdf
Thanks for https://github.com/apourchot/CEM-RL

Đồ án CS211: AI++

Nhớ tải file requirement.txt 

Để có thể training thì nên dùng colab (chạy train.ypnb)

Dùng test.py để test môi trường (nhớ chỉnh file path actor.pkl)


Without importance mixing:
```console
python es_grad.py --env ENV_NAME --use_td3 --output OUTPUT_FOLDER
```

With importance mixing:
```console
python es_grad_im.py --env ENV_NAME --use_td3 --output OUTPUT_FOLDER
```

TD3:
```console
python distributed.py --env ENV_NAME --use_td3 --output OUTPUT_FOLDER
```
