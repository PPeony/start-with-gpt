from langgraph.graph import StateGraph, START,END

def supermarket(state):
    print("supermarket")
    return {"ret": "{}买到了".format(state["ingredients"])}
# state 的值只能保存在一个节点上，下一个节点是新的state，是上一个节点的返回值，所以如果要传值，需要把要传的值作为节点的输出
def recipe(state):
    print("recipe")
    return {"ret": "搜到了菜谱"}

def cooking(state):
    print("cooking")
    return {"ret": "做了一道菜"}

if __name__ == "__main__":
    sg = StateGraph(dict)

    # 定义节点
    sg.add_node("supermarket", supermarket)
    sg.add_node("recipe", recipe)
    sg.add_node("cooking", cooking)

    # 定义起始边
    sg.add_edge(START, "supermarket")

    # 定义普通边
    sg.add_edge("supermarket", "recipe")
    sg.add_edge("recipe", "cooking")

    # 定义结束边
    sg.add_edge("cooking", END)

    graph = sg.compile()
    ret = graph.invoke({"ingredients": "羊排"})

    print(ret)