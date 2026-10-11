"""检索策略选择器。

让大模型根据查询特点，从 4 种检索增强策略里挑一个：
1. 直接检索       —— 意图明确、要查特定信息（绝大多数医疗问题）
2. 假设问题检索   —— 抽象问题，先生成假设答案再检索（HyDE）
3. 子查询检索     —— 多实体/多方面对比类问题
4. 回溯问题检索   —— 冗长复杂，先简化再检索

失败/无法判定时统一回退「直接检索」，保证主流程永远可用。
"""
from langchain_core.prompts import PromptTemplate

from app.config import get_settings
from app.core import llm
from app.core.logger import get_logger

logger = get_logger(__name__)

# 合法策略名（与 dispatch 逻辑对应）
STRATEGY_DIRECT = "直接检索"
STRATEGY_HYDE = "假设问题检索"
STRATEGY_SUBQUERY = "子查询检索"
STRATEGY_BACKTRACKING = "回溯问题检索"


class StrategySelector:
    """基于 LLM 的检索策略选择器。"""

    def __init__(self) -> None:
        self.template = self._get_strategy_prompt()

    @staticmethod
    def _get_strategy_prompt() -> PromptTemplate:
        """策略选择提示词模板（few-shot 引导，示例改写为糖尿病随访场景）。"""
        return PromptTemplate(
            template="""
            你是一个智能助手，负责分析用户查询 {query}，并从以下四种检索增强策略中选择一个最适合的策略，直接返回策略名称，不需要解释过程。

            以下是几种检索增强策略及其适用场景：

            1.  **直接检索：**
                * 描述：对用户查询直接进行检索，不进行任何增强处理。
                * 适用场景：查询意图明确，需要从知识库中检索**特定信息**的问题，例如：
                    * 查询：空腹血糖的控制目标是多少？
                    * 策略：直接检索
                    * 查询：二甲双胍应该怎么吃？
                    * 策略：直接检索
            2.  **假设问题检索（HyDE）：**
                * 描述：使用 LLM 生成一个假设的答案，然后基于假设答案进行检索。
                * 适用场景：查询较为抽象，直接检索效果不佳的问题，例如：
                    * 查询：糖尿病患者的日常饮食需要注意哪些方面？
                    * 策略：假设问题检索
            3.  **子查询检索：**
                * 描述：将复杂的用户查询拆分为多个简单的子查询，分别检索并合并结果。
                * 适用场景：查询涉及多个实体或方面，需要分别检索不同信息的问题，例如：
                    * 查询：比较胰岛素和口服降糖药的优缺点。
                    * 策略：子查询检索
            4.  **回溯问题检索：**
                * 描述：将复杂的用户查询转化为更基础、更易于检索的问题，然后进行检索。
                * 适用场景：查询较为复杂，需要简化后才能有效检索的问题，例如：
                    * 查询：我是一名 10 年病程的 2 型糖尿病患者，最近血糖控制不好，晚上还出现低血糖，我该怎么办？
                    * 策略：回溯问题检索

            根据用户查询 {query}，直接返回最适合的策略名称，例如 "直接检索"。不要输出任何分析过程或其他内容。
            """,
            input_variables=["query"],
        )

    def select_strategy(self, query: str) -> str:
        """选择检索策略，返回策略名（异常时回退「直接检索」）。"""
        if not get_settings().enable_strategy_selector:
            return STRATEGY_DIRECT
        try:
            prompt = self.template.format(query=query)
            strategy = llm.generate(prompt, system="你是一个有用的助手，能够根据用户输入的 Prompt 严格执行并返回可靠的结果").strip()
            # 归一化：模型可能返回带引号/多余字符，只认 4 种合法值
            for name in (STRATEGY_HYDE, STRATEGY_SUBQUERY, STRATEGY_BACKTRACKING, STRATEGY_DIRECT):
                if name in strategy:
                    logger.info("strategy.selected | query=%s | strategy=%s", query[:30], name)
                    return name
            return STRATEGY_DIRECT
        except Exception as e:
            logger.error("strategy.select_failed | error=%s", e)
            return STRATEGY_DIRECT
