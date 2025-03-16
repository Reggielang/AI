# 上课笔记

## Apifox和Postman
### OpenAI接口文档
官方文档：https://platform.openai.com/docs/api-reference/chat/create
中文文档：https://apifox.com/apidoc/shared-012b355c-5a9e-4b61-aeca-105d78dc51d5?pwd=jkai
导入到apifox：导入OpenAI.apifox.json 或者页面克隆，自动生成python代码
导入到posfman：导入OpenAI.postman.json

## OpenAI环境变量
### Windows 导入环境变量

```powershell
#HK代理环境，不需要科学上网(价格便宜、有安全风险，适合个人开发调试)
setx OPENAI_BASE_URL "https://api.openai-hk.com/v1"
setx OPENAI_API_KEY "hk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

#官方环境，需要科学上网(价格高、安全可靠，适合个人企业生产部署)
setx OPENAI_BASE_URL "https://api.openai.com/v1"
setx OPENAI_API_KEY "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

注意：每次执行完，需要重启PyCharm才能生效
```

### Mac 导入环境变量

```bash
#HK代理环境，不需要科学上网
export OPENAI_API_KEY='https://api.openai-hk.com/v1'
export OPENAI_API_KEY='hk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

#官方环境，需要科学上网
export OPENAI_API_KEY='https://api.openai.com/v1'
export OPENAI_API_KEY='sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
```

