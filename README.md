## Zssm(core) 插件说明

一个为 AstrBot 提供「知识说明 / 消息解释」能力的插件。支持文本、图片以及 QQ 群文件（含 PDF，支持转 Markdown），可智能理解被回复的消息或内容链接，并输出结构化的中文解释。

---

## 功能概览

### 文本解释
- 指令 `/zssm` 或关键词 `zssm` 触发，对携带的文本进行简洁说明。
- 支持「zssm + 文本」直接解释当前消息文本，无需回复。

### 图片 / 图文消息解释
- 支持发送消息时附带图片进行解释。
- 支持回复一条包含图片的消息后发送 `zssm`，自动提取上下文文本和图片 URL，调用具备图片能力的多模态模型进行转述。
- 支持 Napcat/OneBot 仅提供 `message_id` 的场景，通过 `get_msg` 回溯原消息。

### 联网搜索
- 使用「zssm + 内容」格式可进行联网搜索（需配置 Search Provider）。
- 如果同时回复了一条消息，被回复的内容将作为上下文信息一并发送给模型。

### QQ 群文件解释（含 PDF→Markdown）
- 回复 QQ 群文件消息后发送 `zssm`：
  - 对文本类扩展名（默认 `txt,md,log,json,csv,ini,cfg,yml,yaml,py` 等）读取前若干 KB 内容作为预览，连同文件名/说明一起交给 LLM 解释。
  - 对 PDF 文件：
    - 使用 PyMuPDF（`fitz`）优先将 PDF 内容转换为 Markdown（`page.get_text("markdown")`），保留标题、列表等基本格式；
    - 若 PyMuPDF 不可用，则回退到 PyPDF2 提取纯文本，并做段落归一化；
    - 根据配置的最大大小限制，仅在文本长度合理时追加到解释上下文中。
- 支持 Napcat 的 QQ 合并转发聊天记录中嵌套的群文件：
  - 通过 `get_forward_msg` 拉取转发节点，在节点内容中查找首个群文件并进行同样的预览与解释。

### QQ 合并转发聊天记录解释
- 对于 QQ 合并转发（`forward` / `Node` / `Nodes` / `get_forward_msg`）：
  - 先展开所有节点，汇总文本与图片，作为「整段聊天记录」来解释。

### 多模型 / 多模态支持
- 文本解释可指定专用 Provider（`text_provider_id`）。
- 图片解释优先使用支持图片输入的 Provider（`image_provider_id`）。
- 联网搜索可指定专用 Provider（`search_provider_id`）。
- 所有调用在失败时均带有回退逻辑（例如回退到当前会话 Provider）。

### 人格覆写
- 支持通过配置开关 `persona_override_enabled` 控制是否使用自定义人格设定。
- 开启后，将使用 `persona_setting` 中的内容作为系统提示词。
- 关闭时，使用插件内置的默认提示词。

---

## 触发方式

### 指令触发

- `/zssm`
  - 最基本用法：`/zssm` + 回复消息，解释被回复的文本/图片/群文件/合并转发等。
  - 或 `/zssm 这段命令是干什么的？`，直接解释当前消息中的文本。
  - 或 `/zssm [图片]`，直接解释当前消息中图片的含义。
  - 或 `/zssm 什么是量子计算`，进行联网搜索并回答。

### 关键词触发

- 文本中包含关键字 `zssm`（忽略大小写、常见前缀、Reply/At 噪音）时自动触发：
  - `zssm 这条报错什么意思`
  - `@Bot zssm 请解释上面这段话`
  - 若以 `/zssm` 开头则优先视为指令，不会重复触发关键词逻辑。
- 该行为可通过配置项 `enable_keyword_zssm` 关闭。

### 回复场景

- 回复任意消息后发送 `zssm` / `/zssm`：
  - 若被回复中包含文本/图片，则优先解释文字和图片；
  - 若检测到 Napcat 群文件，则启动文件内容预览；
  - 若是 QQ 合并转发，则会自动抓取所有节点内容进行整段解释；
  - 若以上都无法获取有效内容，则提示「请输入要解释的内容」。

---

## 配置项说明

| 配置项 | 类型 | 默认值 | 说明 |
|--------|------|--------|------|
| `persona_override_enabled` | bool | `false` | 是否启用人格覆写。开启后使用下方的人格设定，关闭则使用内置默认提示词。 |
| `persona_setting` | text | (内置提示词) | 自定义系统提示词（仅在人格覆写开启时生效）。 |
| `enable_keyword_zssm` | bool | `true` | 是否启用"zssm"关键词自动触发。关闭后仅响应 `/zssm` 指令。 |
| `llm_timeout_sec` | int | `90` | LLM 调用超时（秒），范围建议 5-600。 |
| `file_preview_max_size_kb` | int | `100` | 群文件内容预览允许的最大文件大小（KB）。超过该大小仅展示元信息。 |
| `file_preview_exts` | string | `txt,md,log,...` | 群文件内容预览的文本扩展名（逗号分隔）。 |
| `text_provider_id` | string | (空) | 文本解释优先使用的 Provider ID。留空则使用当前会话 Provider。 |
| `image_provider_id` | string | (空) | 图片转述优先使用的 Provider ID。留空则自动选择具备图片能力的 Provider。 |
| `search_provider_id` | string | (空) | 联网搜索优先使用的 Provider ID。用于「zssm + 内容」搜索模式。 |

---

## 依赖

- **必需**：`aiohttp` - 用于异步网络请求。
- **可选**：
  - `PyMuPDF`（`fitz`）- 用于 PDF 转 Markdown，效果更佳。
  - `PyPDF2` - PyMuPDF 不可用时的备选 PDF 解析库。

---

## 提示词与输出格式

- 系统提示词与用户提示词模板集中在 `prompt_utils.py`：
  - `DEFAULT_SYSTEM_PROMPT`：约束 LLM 输出结构，如「关键词行 + 总结 + **详细阐述**」。
  - `DEFAULT_TEXT_USER_PROMPT` / `DEFAULT_IMAGE_USER_PROMPT`：分别用于纯文本、图文场景。
- 如需自定义输出格式，可通过配置项 `persona_setting` 进行覆写（需开启 `persona_override_enabled`）。

---

## 已知限制与 TODO

- 对 PDF 中的表格、公式等结构，PyMuPDF 的 Markdown 输出虽有改善，但不保证 100% 还原。
- 部分复杂网站（强 JS 渲染、登录态依赖）可能需要手动截图或粘贴正文。

---

## 更新日志

### v1.1.0
- 新增「人格覆写开关」配置项，可灵活切换默认提示词与自定义人格。
- 移除视频和语音解析功能，精简插件体积。
- 修复回复消息时多余 `@` 符号的问题。
- 优化 PDF 解析，将 CPU 密集型操作移至线程池执行。

### v1.0.0
- 初始版本发布。
- 支持文本、图片、群文件（含 PDF）解释。
- 支持 QQ 合并转发聊天记录解释。
- 支持联网搜索模式。

---

## 特别感谢

- [Reina](https://github.com/Ri-Nai) - 本插件参考了其 JSON 消息处理代码并完善了 JSON 卡片消息的处理。
- [氕氙](https://github.com/piexian) - 感谢稀有气体同学的 PR。
- 原作者 [薄暝](https://github.com/xiaoxi68) - 原始插件 `astrbot_zssm_explain` 的开发者。
