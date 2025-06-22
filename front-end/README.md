# 🧠 Retina AI Diagnosis System

基于深度学习的眼底图像智能诊断平台，支持本地快速部署运行。项目使用 Spring Boot 与 MySQL 集成，通过 JPA 自动建库建表，可快速用于开发与演示。

---

## 🚀 快速启动指南

### 1️⃣ 克隆项目

```bash
git clone https://github.com/your-username/retina-ai-diagnosis-system.git
cd retina-ai-diagnosis-system
```


### 2️⃣ 修改数据库配置

配置文件路径：

```
src/main/resources/application.properties
```

#### ⚙️ 参数说明：

| 配置项 | 说明 |
|--------|------|
| `spring.datasource.url` | 数据库连接地址 |
| `createDatabaseIfNotExist=true` | 启动时自动创建数据库 |
| `spring.datasource.username/password` | 本地 MySQL 登录信息 |


### 3️⃣ 启动本地 MySQL 数据库（macOS 示例）

```bash
sudo /usr/local/mysql/support-files/mysql.server start
```

- 默认监听端口：3306
- 默认用户名：`root`
- 默认密码：根据你设置或重置后的值（如 `123456`）


### 4️⃣ 启动项目

直接点击 IDE 中的 LoginDemoApplication 运行按钮
打开网址：

```
http://localhost:8080
```
or
```
http://localhost:8080/login
```


## 🗂 项目结构概览

```
retina-ai-diagnosis-system/
├── src/
│   └── main/
│       ├── java/                       # Java 源码
│       │   └── .../                   # 包含 controller, service, model 等模块
│       └── resources/
│           ├── application.properties # 配置文件（需本地修改）
│           ├── static/                # 前端静态资源
│           └── templates/             # HTML 模板文件（Thymeleaf）
├── pom.xml                            # Maven 配置文件
├── README.md                          # 项目说明文档
```


## 🧰 技术栈

- **Java 17+**
- **Spring Boot 3.x**
- **Spring Data JPA + Hibernate**
- **MySQL 8.x**
- **Thymeleaf**
- **Maven Wrapper**
- **Navicat / TablePlus / DBeaver 等数据库客户端**

##常见问题 FAQ

###Navicat 报错 2002 无法连接数据库？

确保 MySQL 开启了 TCP 连接功能：

```ini
[mysqld]
bind-address = 127.0.0.1
```

重启服务：

```bash
sudo /usr/local/mysql/support-files/mysql.server restart
```

### Spring Boot 启动时报数据库连接失败？

请确认：

- MySQL 已运行
- `application.properties` 中账号密码无误
- 使用 Navicat 可连接数据库
