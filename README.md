# odir-classification 眼底图像智能诊断平台

本项目是一个基于深度学习的眼底图像智能诊断平台，支持本地快速部署与运行。平台集成了深度学习模型和前后端服务，能够对眼底图像进行智能分类，识别八种不同的眼部疾病。

## 项目总览

```
odir-classification/
├── README.md # 项目总说明文档
├── ai-model-python/ # AI 模型及相关脚本 (Python)
└── front-end/ # 后端服务和前端页面 (Java Spring Boot)
```

## AI 模型部分

该部分基于 MindSpore 和 MindCV 框架，旨在实现对眼底镜图像的智能分类，能够识别八种不同的眼部疾病。项目利用深度学习技术，通过训练 ResNet50 模型来分析眼底图像，并自动区分正常、白内障、糖尿病、青光眼、高血压、近视等多种眼部病理状况。

### 技术栈

* MindSpore
* MindCV
* Python 3.11

### 配置与训练

详见 [README_CN_ai-model-python.md](ai-model-python/README_CN_ai-model-python.md)

## 前后端服务部分

该部分使用 Spring Boot 与 MySQL 集成，通过 JPA 自动建库建表，可快速用于开发与演示.

### 技术栈

* Java 17+
* Spring Boot 3.x
* Spring Data JPA + Hibernate
* MySQL 8.x
* Thymeleaf
* Maven Wrapper
* Navicat / TablePlus / DBeaver 等数据库客户端

### 配置与运行

详见 [README_CN_front-end.md](front-end/README_CN_front-end.md)