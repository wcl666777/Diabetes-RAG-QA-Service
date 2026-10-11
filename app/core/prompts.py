"""RAG 问答 Prompt 模板。

统一管理 RAG 全流程所需模板：
- rag_prompt           最终回答模板（含历史 / 上下文 / 问题 / 客服电话）
- hyde_prompt          假设问题检索（HyDE）
- subquery_prompt      子查询检索
- backtracking_prompt  回溯问题检索

医疗场景额外要求：回答必须有据、标注来源、拿不准时引导转人工。
"""
from langchain_core.prompts import PromptTemplate


def rag_prompt() -> PromptTemplate:
    """最终回答模板：融合对话历史 + 检索上下文 + 问题。"""
    return PromptTemplate(
        template=(
            "你是一名专业的糖尿病随访管理助手，负责解答糖尿病出院患者的咨询。\n"
            "请参考对话历史和检索到的上下文回答问题，并严格遵守以下规则：\n"
            "1. 优先基于【检索到的资料】回答，不要编造医学事实；\n"
            "2. 回答中应引用资料来源（用【来源：xxx】标注）；\n"
            "3. 如果资料不足或问题涉及高风险医疗决策（如调整用药、急性重症），\n"
            "   必须明确提示“建议咨询医生”，不要给出确定性处方建议；\n"
            "4. 使用通俗易懂的中文回答患者。\n\n"
            "【对话历史】\n{history}\n\n"
            "【检索到的资料】\n{context}\n\n"
            "【患者问题】\n{question}\n\n"
            "如果资料不足无法回答，请回复：“信息不足，无法回答，建议咨询您的主管医生，"
            "或拨打随访电话：{phone}。”\n\n"
            "【回答】"
        ),
        input_variables=["context", "history", "question", "phone"],
    )


def hyde_prompt() -> PromptTemplate:
    """假设问题检索：先生成一个简短的假设答案，再用它去检索（提升抽象问题的召回）。"""
    return PromptTemplate(
        template=(
            "假设你是患者，想了解以下问题，请基于医学常识生成一个简短的假设答案：\n"
            "问题: {query}\n"
            "假设答案:"
        ),
        input_variables=["query"],
    )


def subquery_prompt() -> PromptTemplate:
    """子查询检索：把复杂问题拆成多个简单子查询，逐行返回。"""
    return PromptTemplate(
        template=(
            "将以下复杂查询分解为多个简单子查询，每行一个子查询，不要编号：\n"
            "查询: {query}\n"
            "子查询:"
        ),
        input_variables=["query"],
    )


def backtracking_prompt() -> PromptTemplate:
    """回溯问题检索：把冗长复杂的问题简化为一个更易检索的问题。"""
    return PromptTemplate(
        template=(
            "将以下复杂查询简化为一个更简单、更便于检索的问题：\n"
            "查询: {query}\n"
            "简化问题:"
        ),
        input_variables=["query"],
    )
