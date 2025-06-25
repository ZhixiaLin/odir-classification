# odir-classification front-end

基于深度学习的眼底图像智能诊断平台，支持本地快速部署运行。项目使用 Spring Boot 与 MySQL 集成，通过 JPA 自动建库建表，可快速用于开发与演示。

---

## 项目配置

### 1. 克隆项目

```bash
git clone this-repo-url.git
```

### 2. 修改数据库配置

配置文件路径：
```
src/main/resources/application.properties
```

参数说明：

| 配置项 | 说明 |
|--------|------|
| `spring.datasource.url` | 数据库连接地址 |
| `createDatabaseIfNotExist=true` | 启动时自动创建数据库 |
| `spring.datasource.username/password` | 本地 MySQL 登录信息 |


### 3. 启动本地 MySQL 数据库（macOS 示例）

```bash
sudo /usr/local/mysql/support-files/mysql.server start
```

- 默认监听端口：3306
- 默认用户名：`root`
- 默认密码：根据你设置或重置后的值（如 `123456`）

### 4. 配置并运行项目
1. 使用 IDEA 打开项目根目录  
2. 将 `front-end/src` 标记为 `Sources Root`  
3. 右键 `front-end/pom.xml` → `Add as Maven Project`  


## 启动项目

直接点击 IDEA 中的 `LoginDemoApplication` 运行按钮  
打开网址：

```
http://localhost:8080
```
或
```
http://localhost:8080/login
```

## 项目结构概览

```
odir-classification/
├── README.md                      # 项目总说明文档
├── ai-model-python/               # AI 模型及相关脚本 (Python)
│
└── front-end/                     # 后端服务和前端页面 (Java Spring Boot)
    ├── pom.xml                    # Maven 项目配置文件
    ├── README_CN_front-end.md     # 前端项目说明文档
    └── src/
        └── main/
            ├── java/              # Java 核心代码
            └── resources/
                ├── application.properties # Spring Boot 应用程序配置
                ├── static/        # 存放 CSS, JavaScript, 图片等静态文件
                └── templates/     # 存放 Thymeleaf HTML 页面模板
```


## 技术栈

- Java 17+
- Spring Boot 3.x
- Spring Data JPA + Hibernate
- MySQL 8.x
- Thymeleaf
- Maven Wrapper
- Navicat / TablePlus / DBeaver 等数据库客户端

## 常见问题 FAQ

### Navicat 报错 2002 无法连接数据库？

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
