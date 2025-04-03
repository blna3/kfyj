import json
import os
import re
from concurrent.futures import ThreadPoolExecutor
from openai import OpenAI
from typing import List, Dict, Any

# 配置API客户端
client = OpenAI(
    api_key="MveHKMMpioqCeKigl2_YUObSeEhuFkd5r1pCd8uhcbk",
    base_url="https://zhenze-huhehaote.cmecloud.cn/inference-api/exp-api/inf-1336781912337387520/v1"
)

def extract_entities(text: str) -> (List[Dict[str, str]], str): # type: ignore
    """使用Deepseek API进行命名实体识别，返回结果和原始响应"""
    try:
        response = client.chat.completions.create(
            model="default",
            messages=[
                {
                    "role": "system",
                    "content": (
                        "你是一个专业的命名实体识别助手，需要从文本中识别实体名称和实体类型。"
                        "请严格使用JSON格式输出结果，示例：[{\"名称\": \"...\", \"类型\": \"...\"}]。"
                        "确保使用基础或基本的类型,如人物,职业,时间,文件,编码等,避免使用更具体的术语"
                        "如果未识别到实体请返回空数组。不要包含任何额外内容。"
                        "请确保始终包含有效的JSON代码块。"
                    )
                },
                {"role": "user", "content": f"请从以下文本中提取命名实体：{text}"}
            ],
            max_tokens=204800,
            temperature=0.3,
            stream=False
        )

        raw_output = response.choices[0].message.content
        json_match = re.search(r'```json\n(.*?)\n```', raw_output, re.DOTALL)
        
        if json_match:
            try:
                return json.loads(json_match.group(1)), raw_output
            except json.JSONDecodeError:
                raise ValueError(f"无效的JSON格式: {json_match.group(1)}")
        return [], raw_output
        
    except Exception as e:
        raise RuntimeError(f"API调用失败: {str(e)}") from e

def process_content(content: str) -> (List[Dict[str, str]], str): # type: ignore
    """处理单个content内容，返回结果和原始响应"""
    if not content:
        return None, None
    
    try:
        entities, raw_output = extract_entities(content)
        return entities, raw_output
    except Exception as e:
        error_msg = f"实体识别失败: {str(e)}"
        if hasattr(e, "raw_output"):
            error_msg += f"\n原始响应: {e.raw_output}"
        raise RuntimeError(error_msg) from e

def process_file(file_path: str):
    """处理单个JSON文件"""
    try:
        with open(file_path, 'r+', encoding='utf-8') as f:
            data = json.load(f)
            original_position = f.tell()
            
            def process_sections(sections):
                for section in sections:
                    if 'content' in section:
                        try:
                            # 保留原始内容用于错误处理
                            original_content = section.get('content')
                            
                            # 处理并记录原始输出
                            ner_result, raw_output = process_content(original_content)
                            section['NER'] = ner_result
                            
                            # 添加调试信息
                            section['_debug'] = {
                                'api_raw_output': raw_output,
                                'processed': True
                            }
                        except Exception as e:
                            print(f"处理文件 {file_path} 时出错：{str(e)}")
                            section['NER'] = None
                            section['_debug'] = {
                                'error': str(e),
                                'raw_content': original_content,
                                'processed': False
                            }
                    if 'subsections' in section:
                        process_sections(section['subsections'])

            process_sections(data['sections'])
            
            # 回写文件时保留原始格式
            f.seek(0)
            json.dump(data, f, ensure_ascii=False, indent=2)
            f.truncate()
            
        print(f"成功处理文件：{file_path}")
        
    except Exception as e:
        print(f"处理文件 {file_path} 时发生严重错误：{str(e)}")

def main(folder_path: str):
    """主处理函数"""
    json_files = [
        os.path.join(root, file)
        for root, _, files in os.walk(folder_path)
        for file in files if file.endswith('.json')
    ]

    # 使用带错误处理的线程池
    with ThreadPoolExecutor(max_workers=4) as executor:
        futures = []
        for file in json_files:
            try:
                futures.append(executor.submit(process_file, file))
            except Exception as e:
                print(f"任务提交失败 {file}: {str(e)}")
        
        for future in futures:
            try:
                future.result()
            except Exception as e:
                print(f"任务执行失败: {str(e)}")

if __name__ == "__main__":
    folder_path = r"E:\Desktop\10"  # 替换为实际路径
    main(folder_path)