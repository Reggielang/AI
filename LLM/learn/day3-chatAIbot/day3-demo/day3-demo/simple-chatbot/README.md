

## Windows 导入环境变量

```powershell
#HK代理环境，不需要科学上网(推荐)
setx OPENAI_BASE_URL "https://api.openai-hk.com/v1"
setx OPENAI_API_KEY "hk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

#官方环境，需要科学上网
setx OPENAI_BASE_URL "https://api.openai.com/v1"
setx OPENAI_API_KEY "sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx"

注意：每次执行完，需要重启PyCharm才能生效
```



## Mac 导入环境变量

```bash
#HK代理环境，不需要科学上网(推荐)
export OPENAI_API_KEY='https://api.openai-hk.com/v1'
export OPENAI_API_KEY='hk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'

#官方环境，需要科学上网
export OPENAI_API_KEY='https://api.openai.com/v1'
export OPENAI_API_KEY='sk-xxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxxx'
```

