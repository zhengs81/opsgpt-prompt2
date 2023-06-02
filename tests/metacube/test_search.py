"""测试metacube数据的搜索功能

主要是实例和对象的精准搜索和模糊搜索
"""

def test_instance_exact_search(search_agent_executor):
    """通过实例编码`APPCC-000001`精准搜索实例信息"""

    question = "APPCC-000001是什么？"
    reply = search_agent_executor.run(question)

    mentioned_instance_type_code = "APPCC" in reply
    mentioned_instance_code = "APPCC-000001" in reply
    mentioned_entity_type = "实例" in reply

    # mentioned all above
    assert(all([
        mentioned_entity_type,
        mentioned_instance_code,
        mentioned_instance_type_code
        ]))


def test_instance_fuzz_search(search_agent_executor):
    """通过有输入错误的实例编码`APPCC-0001`模糊搜索实例信息"""

    question = "APPCC-0001是什么？"
    reply = search_agent_executor.run(question)

    mentioned_instance_type_code = "APPCC" in reply
    mentioned_instance_code = "APPCC-000001" in reply
    mentioned_entity_type = "实例" in reply

    # mentioned all above
    assert(all([
        mentioned_entity_type,
        mentioned_instance_code,
        mentioned_instance_type_code
        ]))


def test_object_exact_search(search_agent_executor):
    """通过对象编码`Service`精准搜索对象信息"""
    question = "Service是什么？"
    reply = search_agent_executor.run(question)
    
    mentioned_instance_type_code = "Service" in reply
    mentioned_entity_type = "对象" in reply

    # mentioned all above
    assert(all([
        mentioned_entity_type,
        mentioned_instance_type_code
        ]))


def test_object_fuzz_search(search_agent_executor):
    """通过有输入错误的对象编码`Servce`模糊搜索对象信息"""
    question = "Servce是什么？"
    reply = search_agent_executor.run(question)
    
    mentioned_instance_type_code = "Service" in reply
    mentioned_entity_type = "对象" in reply

    # mentioned all above
    assert(all([
        mentioned_entity_type,
        mentioned_instance_type_code
        ]))


def test_instance_search_by_name(search_agent_executor):
    """通过有输入部分实例名称`200.20.119.9`模糊搜索实例信息"""
    
    question = "200.20.119.9是什么？"
    reply = search_agent_executor.run(question)
    
    mentioned_instance_type_code = "Pod" in reply
    mentioned_instance_code = "62e90690158b8671d0034939" in reply
    mentioned_instance_name = "test-1_200.20.119.9" in reply
    mentioned_entity_type = "实例" in reply

    # mentioned all above
    assert(all([
        mentioned_instance_type_code,
        mentioned_instance_code,
        mentioned_instance_name,
        mentioned_entity_type
        ]))