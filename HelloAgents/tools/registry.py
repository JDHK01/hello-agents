from typing import Dict, Any


class ToolRegistry:
    '''
    工具管理器, 默认包含终止行动(旨在将finish变为action); 所以这里的ToolExecutor其实是ActionExecutor
    '''
    def __init__(self):
        self.tools : Dict[str, Dict[str, Any]] = {}
        self.registerTool(
            name = "finish",
            description="finish(conclusion: str): Call this action when you think it's time to end, args is your conclusion",
            function=None
        )


    def registerTool(self, name:str, description:str, function:Any):
        self.tools[name] = {
            "description": description,
            "function":function
        }


    def useTool(self, name):
        return self.tools.get(name).get("function")
    

    def introduceTool(self):
        return '\n'.join(
            [
                f'- {info.get("description")}' for info in self.tools.values()
            ]
        )