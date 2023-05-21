import re


def test_instance_exact_search(search_agent_executor):
    """通过实例编码`APPCC-000001`精准搜索实例信息"""
    object_pattern = re.compile(
        r".*[^0-9a-zA-Z]APPCC[^0-9].*"
    )
    
    instance_pattern = re.compile(
        r".*[^0-9a-zA-Z]APPCC-000001[^0-9].*"
    )

    type_pattern = re.compile(
        r".*实例.*"
    )
    
    respond = search_agent_executor.run(
        "APPCC-000001是什么？"
    )

    assert bool(object_pattern.match(respond))
    assert bool(instance_pattern.match(respond))
    assert bool(type_pattern.match(respond))


def test_instance_fuzz_search(search_agent_executor):
    """通过有输入错误的实例编码`APPCC-0001`模糊搜索实例信息"""
    object_pattern = re.compile(
        r".*[^0-9a-zA-Z]APPCC[^0-9].*"
    )
    
    instance_pattern = re.compile(
        r".*[^0-9a-zA-Z]APPCC-000001[^0-9].*"
    )

    type_pattern = re.compile(
        r".*实例.*"
    )
    
    respond = search_agent_executor.run(
        "APPCC-0001是什么？"
    )

    assert bool(object_pattern.match(respond))
    assert bool(instance_pattern.match(respond))
    assert bool(type_pattern.match(respond))


def test_object_exact_search(search_agent_executor):
    """通过实例编码`APPCC-000001`精准搜索实例信息"""
    object_pattern = re.compile(
        r".*[^0-9a-zA-Z]Service[^0-9a-zA-Z].*"
    )
    
    respond = search_agent_executor.run(
        "Service是什么？"
    )

    type_pattern = re.compile(
        r".*对象.*"
    )

    assert bool(object_pattern.match(respond))
    assert bool(type_pattern.match(respond))


def test_object_fuzz_search(search_agent_executor):
    """通过有输入错误的实例编码`APPCC-0001`模糊搜索实例信息"""
    object_pattern = re.compile(
        r".*[^0-9a-zA-Z]Service[^0-9a-zA-Z].*"
    )
    
    respond = search_agent_executor.run(
        "Servce是什么？"
    )

    type_pattern = re.compile(
        r".*对象.*"
    )

    assert bool(object_pattern.match(respond))
    assert bool(type_pattern.match(respond))


def test_instance_search_by_name(search_agent_executor):
    """通过有输入部分实例名称`200.20.119.9`模糊搜索实例信息"""
    object_pattern = re.compile(
        r".*[^0-9a-zA-Z]Pod[^0-9a-zA-Z].*"
    )

    instance_code_pattern = re.compile(
        r".*[^0-9a-zA-Z]62e90690158b8671d0034939[^0-9a-zA-Z].*"
    )

    instance_name_pattern = re.compile(
        r".*[^0-9a-zA-Z]test-1_200.20.119.9[^0-9a-zA-Z].*"
    )
    
    respond = search_agent_executor.run(
        "200.20.119.9是什么？"
    )

    type_pattern = re.compile(
        r".*实例.*"
    )

    assert bool(object_pattern.match(respond))
    assert bool(instance_code_pattern.match(respond))
    assert bool(instance_name_pattern.match(respond))
    assert bool(type_pattern.match(respond))