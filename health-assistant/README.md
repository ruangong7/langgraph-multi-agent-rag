# 🏥 Personal Health Assistant

> 基于 LangGraph 的多智能体个人健康助手，替代原旅游客服系统，保留完整的多 Agent 架构。

## 项目结构

```
.
├── customer_support_chat/    # 原项目（旅游客服多Agent系统）
├── health-assistant/         # 🆕 个人健康助手
│   ├── customer_support_chat/app/
│   │   ├── core/             # MySQL连接池、状态管理、配置
│   │   ├── services/
│   │   │   ├── assistants/   # 9个专业Agent
│   │   │   ├── tools/        # 健康领域工具函数
│   │   │   ├── guardrails/   # 安全护栏（越狱检测+医疗相关性过滤）
│   │   │   ├── knowledge_graph/  # 医学知识图谱
│   │   │   └── vectordb/     # Advanced RAG + 评估器
│   │   └── graph.py          # LangGraph StateGraph（主图）
│   ├── tests/                # 43个单元测试
│   ├── setup_database.py     # MySQL建库建表脚本
│   └── requirements.txt
└── README.md
```

## 9 个 Agent 说明

| Agent | 功能 | 触发场景 |
|-------|------|----------|
| 🩺 Primary | 意图理解、路由分派 | 所有消息的入口 |
| 📅 Appointment | 预约挂号、查预约、取消 | "帮我挂个协和心内科的号" |
| 💊 Medication | 用药管理、相互作用检查 | "阿莫西林和布洛芬能一起吃吗" |
| 🚨 Emergency | 急救指导 | "我胸口疼得厉害" |
| 🏃 Health Tips | 运动、饮食、睡眠、心理 | "怎么减肥比较健康" |
| 📋 Medical Record | 病历查看、添加 | "查我上次的血常规结果" |
| 🔬 Health Assessment | 症状评估、风险分级 | "我头疼三天了要不要紧" |
| 📚 Medical KB | 疾病知识、药品信息 | "高血压是什么" |
| 🧬 Medical KG | 知识图谱多跳推理 | "我的症状可能是什么病" |

## 技术栈

```
LangGraph StateGraph (Supervisor Pattern)
    ↓
Primary Health Assistant (Orchestrator)
    ↓ 分派到
Appointment / Medication / Emergency / Health Tips
Medical Record / Health Assessment / Medical KB / Medical KG
    ↓ 使用
MySQL (6 tables) + Qdrant (vector RAG) + NetworkX (knowledge graph)
```

- **LLM**: OpenAI 兼容 API（GPT-4o / DeepSeek / Claude 任意切换）
- **数据库**: MySQL 8.0+（连接池，6 表）
- **向量检索**: Qdrant
- **知识图谱**: NetworkX（本地）+ Neo4j 适配器（可选）
- **安全**: Jailbreak 检测 + 医疗相关性过滤 + 敏感操作人工确认
- **测试**: pytest，43 个测试，0.26s 全通过

---

## 🚀 启动指南

### 1. 环境要求

```bash
Python 3.10+
MySQL 8.0+
```

### 2. 安装依赖

```bash
cd health-assistant
pip install -r requirements.txt
```

### 3. 配置 MySQL

```bash
# 方式A：Docker 快速启动 MySQL
docker run -d --name mysql-health \
  -e MYSQL_ROOT_PASSWORD=root123 \
  -e MYSQL_DATABASE=health_assistant \
  -e MYSQL_USER=health_assistant \
  -e MYSQL_PASSWORD=health_pass \
  -p 3306:3306 \
  mysql:8.0

# 方式B：使用已有 MySQL，创建数据库
mysql -u root -p -e "CREATE DATABASE IF NOT EXISTS health_assistant CHARACTER SET utf8mb4;"
mysql -u root -p -e "CREATE USER IF NOT EXISTS 'health_assistant'@'%' IDENTIFIED BY 'health_pass';"
mysql -u root -p -e "GRANT ALL PRIVILEGES ON health_assistant.* TO 'health_assistant'@'%'; FLUSH PRIVILEGES;"
```

### 4. 配置环境变量

```bash
# 必需
export OPENAI_API_KEY="sk-你的key"          # OpenAI API Key
export OPENAI_BASE_URL="https://api.openai.com/v1"  # 或用 DeepSeek: https://api.deepseek.com/v1

# MySQL（如用 Docker 默认值则无需改动）
export MYSQL_HOST="localhost"
export MYSQL_PORT="3306"
export MYSQL_USER="health_assistant"
export MYSQL_PASSWORD="health_pass"
export MYSQL_DATABASE="health_assistant"

# 可选
export MODEL_NAME="gpt-4o-mini"              # LLM模型名
export QDRANT_URL="http://localhost:6333"    # Qdrant地址（可选）
```

**使用 DeepSeek 示例：**
```bash
export OPENAI_API_KEY="sk-你的deepseek-key"
export OPENAI_BASE_URL="https://api.deepseek.com/v1"
export MODEL_NAME="deepseek-chat"
```

### 5. 初始化数据库

```bash
# 自动创建 6 张表 + 插入示例用户数据
export MYSQL_ROOT_PASSWORD="root123"   # root 密码用于创建用户
python setup_database.py
```

输出示例：
```
✅ Database 'health_assistant' ensured
✅ User 'health_assistant' ensured with privileges
✅ All tables verified/created
✅ Sample user data inserted
✅ Database setup complete!
```

### 6. 启动服务

```bash
cd health-assistant
uvicorn customer_support_chat.app.main:app --host 0.0.0.0 --port 8000 --reload
```

### 7. 测试调用

```bash
# API 调用
curl -X POST http://localhost:8000/chat \
  -H "Content-Type: application/json" \
  -d '{"user_id": "user_001", "message": "我头疼发烧，怎么办？"}'

# 健康检查
curl http://localhost:8000/health

# API 文档
open http://localhost:8000/docs
```

### 8. 运行测试

```bash
cd health-assistant
python -m pytest tests/ -v
```

输出：
```
tests/test_health_assistant.py::TestHealthModels::test_entity_creation PASSED
tests/test_health_assistant.py::TestMedicalGraphStore::test_multi_hop_path PASSED
tests/test_health_assistant.py::TestEmergencyGuidance::test_all_emergencies_covered PASSED
...43 passed in 0.26s
```

---

## MySQL 数据库表结构

```sql
users                    -- 用户档案（过敏史、慢性病、紧急联系人）
appointments             -- 预约记录（医生、医院、科室、时间）
medications              -- 用药记录（药品名、剂量、频率、提醒时间）
medical_records          -- 病历（诊断/化验/处方/疫苗/手术）
health_assessments       -- 健康评估记录（症状、风险等级、建议）
emergency_contacts       -- 紧急联系人
```

---

## 安全特性

| 层级 | 机制 |
|------|------|
| 输入层 | Jailbreak 检测（拒绝越狱尝试） |
| 领域层 | 医学相关性过滤（非健康问题友好引导） |
| 操作层 | 挂号/用药/病历写入需人工确认 |
| 输出层 | 每答必附「请咨询医生」免责声明 |

---

## License

MIT
