##第三届阿里云安全算法赛


###成绩：

- 线上初赛：Rank5


- 线下决赛：Rank2 (算法成绩排名Rank3,真心比不过植物大佬)


###目录结构:

- input:存放原始数据集,包括train和test集

- data:存放中间特征和中间结果

- final_code:存放代码

- submit:保存最终结果


###代码说明:
- security_3rd_property.py:全局变量设置
- security_3rd_model.py:工程中使用到的模型
- security_3rd_feature.py:生成特征并保存
- security_3rd_code.py:主代码,生成提交结果

运行顺序：security_3rd_model -> security_3rd_feature -> security_3rd_code


###运行条件:
- python3
- numpy, pandas, sklearn, contextlib, xgboost





