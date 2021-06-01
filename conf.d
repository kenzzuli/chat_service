;conf.d
[program:chat_service]
command=/Users/ken/PycharmProjects/nlp/bot/chat_service/run.sh  ;执行的命令
stdout_logfile=/Users/ken/PycharmProjects/nlp/bot/chat_service/log/out.log ;log的位置
stderr_logfile=/Users/ken/PycharmProjects/nlp/bot/chat_service/log/error.log  ;错误log的位置
directory=/Users/ken/PycharmProjects/nlp/bot/chat_service  ;路径
autostart=true  ;是否自动启动
autorestart=true  ;是否自动重启
startretries=10 ;失败的最大尝试次数