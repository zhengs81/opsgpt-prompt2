"""测试异常森林
"""


def test_create_analysis(ticketseer_agent):
    """测试Agent能按照指令创建排障任务"""

    question = "创建一个实例编码为ACP的排障任务，时间范围在1686038972302到1667019482000"
    reply = ticketseer_agent.run(question)
    
    mentioned_question_code = "问题编码" in reply or "故障编码" in reply
    mentioned_success = "成功" in reply
    
    assert all(
        [mentioned_question_code, mentioned_success]
    )


def test_query_analysis_result(ticketseer_agent):
    """测试能否查询已创建的故障编码对应的结果"""
    
    question = "查看一下故障单MAN202306071845140000的根因是什么？"
    reply = ticketseer_agent.run(question)
    
    mentioned_root_cause = "根因" in reply
    mentioned_instance = "实例" in reply and "ALS_Oracle_1" in reply
    mentioned_anomaly_type = "指标异常" in reply
    
    assert all(
        [
            mentioned_root_cause,
            mentioned_instance,
            mentioned_anomaly_type
        ]
    )


def test_query_alert_analysis_result(ticketseer_agent):
    """查询多个告警相关的根因分析结果"""
    
    question = "查看下告警BIZ000001685605535912的根因分析结果"
    reply = ticketseer_agent.run(question)
    mentioned_failure = "INC202306011545380000" in reply and "故障" in reply
    mentioned_rca_report = "根因报告" in reply and "f302fad26b124c749541feb0a966e9ca" in reply
    
    assert all(
        [
            mentioned_failure,
            mentioned_rca_report
        ]
    )

