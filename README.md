# Дисциплина "Методы искусственного интеллекта" (СПбГЭТУ "ЛЭТИ")

## Необходимое ПО

Рекомендуется следующий способ создания Python-окружения для выполнения лабораторных работ:

```
$ conda create -n aim python=3.9
$ conda activate aim
$ conda install jupyter matplotlib numpy owlready2 gymnasium pygame scikit-fuzzy scikit-learn
$ conda install torch torchvision -c torch
$ pip install natasha
```

Для выполнения лабораторной работы № 2 по онтологиям необходимо установить редактор онтологий Protege (https://protege.stanford.edu/).
На сайте предлагается web-версия редактора, однако она очень урезана и не позволяет выполнить все задания лабораторной.

Для запуска кода необходимо также установить библиотеку `Owlready2` для Python.

Для выполнения лабораторной работы № 3 понадобятся:

Какой-нибудь редактор байесовских сетей. Рекомендую [GeNIe от BayesFusion](https://download.bayesfusion.com/files.html?category=Academia) - 
он очень функциональный и доступна бесплатная академическая версия.

Следующие библиотеки для Python:

- Gymnasium
- PyGame
- scikit-fuzzy

Для выполнения лабораторной работы № 4 понадобятся:

- PyTorch
- Torchvision

Для выполнения лабораторной работы № 5 понадобятся:

- PyTorch
- scikit-learn
- natasha (https://github.com/natasha/natasha)
