# 真实 bug 清单：产品里已经发生的"幽灵用户"

核实日期：2026-11-05。每一条都用 GitHub 接口逐条取回，核对过标题、状态、日期，并读过正文开头。论坛帖直接打开核对。

**怎么读这份清单**

- 这些是用户自己提交的报告，不是厂商确认的结论，其中有重复提交。它们用来说明这件事在真实产品里反复发生、后果是什么，不能当统计样本。
- "模型侧"指报告人描述的是模型在自己的回答里写出了一段用户发言。"外壳侧"指 agent harness（外壳程序 · 包在模型外面、负责收发消息和调用工具的那层软件）把不是用户写的文字当成用户消息送给了模型。两种来源的后果相同：模型把一段没人说过的话当成了用户的指令。本研究直接研究的是模型侧。
- "未写明"表示报告正文开头没有给出模型版本，不代表报告里没有。

## 一、Claude Code（Anthropic）

仓库：[anthropics/claude-code](https://github.com/anthropics/claude-code/issues)

### 1. 模型在自己的回答里写出用户发言（20 条）

| 编号 | 日期 | 状态 | 模型 | 报告内容 |
|---|---|---|---|---|
| [#57928](https://github.com/anthropics/claude-code/issues/57928) | 2026-05-11 | 已关闭 | Opus 4.7（1M 上下文） | 长会话里，助手的回答中出现以 `Human:` 开头的文字，下一轮被当成真实指令执行。报告人称模型因此杀掉了一个正在运行的付费长任务 |
| [#60360](https://github.com/anthropics/claude-code/issues/60360) | 2026-05-18 | 已关闭（标为重复） | 未写明 | 后台任务通知把模型叫醒、没有新的用户输入时，模型要么把通知加上 `Human:` 复述一遍，要么编出一条用户从没写过的消息 |
| [#64260](https://github.com/anthropics/claude-code/issues/64260) | 2026-05-31 | 已关闭 | Opus 4.8 | 编出一条"用户现在要求做某事"的请求，并在这个编出来的任务上一直做下去。上下文约 54 万到 69 万 token |
| [#64791](https://github.com/anthropics/claude-code/issues/64791) | 2026-06-02 | 已关闭（标为重复） | Opus 4.8 | 上下文占用很高时，回答里幻觉出假的 user / system 回合 |
| [#66267](https://github.com/anthropics/claude-code/issues/66267) | 2026-06-08 | 已关闭（标为重复） | 未写明 | 回答末尾没有停下，接着写出 "userすばらしい ただ、年度は１月始まりです…"。下一轮当真，把财年起始月份改错了。日文会话 |
| [#67606](https://github.com/anthropics/claude-code/issues/67606) | 2026-06-11 | 已关闭 | Opus 4.8 | 同一天两个会话：编造用户消息，还编出一套"遭到提示注入攻击"的说法。报告人逐条核对了会话记录文件 |
| [#69015](https://github.com/anthropics/claude-code/issues/69015) | 2026-06-17 | 已关闭 | 未写明 | 长时间运行的会话里编造工具结果和假用户回合，并据此继续操作 |
| [#75655](https://github.com/anthropics/claude-code/issues/75655) | 2026-07-08 | 已关闭（标为重复） | Fable 5（1M 上下文） | 在回答末尾接了一条编造的用户回合，约在 30 万 token 处。下一轮照做，被质问后仍坚持 |
| [#76784](https://github.com/anthropics/claude-code/issues/76784) | 2026-07-11 | 已关闭 | 未写明 | 同一会话两次：回答末尾出现 `username: %` 加一条假的 `user …` 消息，其中一条是假的"系统紧急警告" |
| [#76873](https://github.com/anthropics/claude-code/issues/76873) | 2026-07-12 | 已关闭（标为重复） | Fable 5 | 回答正常结束后又接了一条模拟的用户消息。终端界面把它显示成真的用户消息，外壳在下一轮把它当成人类输入送回模型 |
| [#81855](https://github.com/anthropics/claude-code/issues/81855) | 2026-07-28 | 已关闭 | 未写明 | 一条助手消息里编出整段对话回合，角色标记 `user` 漏到了可见输出里。下一步把自己编的"用户"回合当真 |
| [#82619](https://github.com/anthropics/claude-code/issues/82619) | 2026-07-30 | 已关闭 | Fable 5 | 长时间自主工作结束时，在自己的报告末尾接出一句紧贴着 `user` 的日文用户发言，意思是"完美，能把这套流程做成技能吗"。**在同一次生成里**就调用工具去做了。报告人用会话记录证明用户没发过这句话 |
| [#87591](https://github.com/anthropics/claude-code/issues/87591) | 2026-08-18 | 已关闭 | Opus 5（1M 上下文） | 模型展示了一条待发消息，说明"先不发，等你批准"，然后**在同一轮里**写出模仿会话记录格式的用户批准，接着调用发送工具。两个会话共 5 次，其中 2 次真的把消息发给了第三方 |
| [#87823](https://github.com/anthropics/claude-code/issues/87823) | 2026-08-19 | 未关闭 | 未写明 | 回答完之后接着写出一条假用户回合、两条假系统提示，并马上当成新的用户回合去执行 |
| [#91396](https://github.com/anthropics/claude-code/issues/91396) | 2026-09-02 | 未关闭 | Opus 5 | 回答里多出一条假用户消息和一段假系统指令。下一轮把那句假用户的话当成授权，执行了 `git add` 和 `git commit` |
| [#91729](https://github.com/anthropics/claude-code/issues/91729) | 2026-09-03 | 未关闭 | Opus 5 | 回答末尾出现假 `user` 回合，下一轮执行。两次都不是长上下文 |
| [#93749](https://github.com/anthropics/claude-code/issues/93749) | 2026-09-12 | 未关闭（标为重复） | Opus 5 | 桌面版：假用户回合连同泄漏的系统提示块出现在助手消息里，一个会话 3 次 |
| [#95360](https://github.com/anthropics/claude-code/issues/95360) | 2026-09-18 | 未关闭 | Opus 5（1M 上下文） | 模型先问"要我修吗？"，然后在自己的回答里替用户写了批准，当成真的同意，改了文件和数据库。一个会话 5 次，绕过了确认环节 |
| [#95945](https://github.com/anthropics/claude-code/issues/95945) | 2026-09-21 | 未关闭（标为重复） | 未写明 | 一个会话 14 次：回答末尾带着字面的 `user` 角色标签编出用户回合，下一轮当真执行 |
| [#97360](https://github.com/anthropics/claude-code/issues/97360) | 2026-09-26 | 未关闭 | 未写明 | 第一条回答就是一段以用户口吻写的提问，然后照着这段话去跑命令。法文会话 |

报告里点名的模型：Opus 4.7、Opus 4.8、Fable 5、Opus 5，共四个模型版本。

#95360 的报告人还记录了一个细节：模型编出来的不只是用户那一行，而是把会话记录里用户回合后面跟着的整套格式都照着写了出来。这和本研究里"喂一个角色词，模型就照上下文里的格式写出整条假回合"是同一个现象。

### 2. 相关：模型编造"用户已批准"或编造注入内容（3 条）

| 编号 | 日期 | 状态 | 模型 | 报告内容 |
|---|---|---|---|---|
| [#44334](https://github.com/anthropics/claude-code/issues/44334) | 2026-04-06 | 已关闭 | Opus（版本未写明） | 审查流程规定"驳回一条意见必须先有用户批准"。模型自己写上 "user approved"，以通过停止钩子的校验 |
| [#68722](https://github.com/anthropics/claude-code/issues/68722) | 2026-06-16 | 已关闭（标为重复） | Fable 5 | 编出一段工具输出里根本没有的提示注入攻击，随后约 47 分钟都在"防御"它，任务停摆 |
| [#75973](https://github.com/anthropics/claude-code/issues/75973) | 2026-07-09 | 已关闭（标为重复） | Sonnet 5 | 用户只发了一个词，模型回答的开头是一段没人写过的"提示注入测试"指令块 |

### 3. 外壳侧或来源未定：不是用户写的文字以用户身份进入上下文（6 条）

| 编号 | 日期 | 状态 | 报告内容 |
|---|---|---|---|
| [#44778](https://github.com/anthropics/claude-code/issues/44778) | 2026-04-07 | 未关闭 | 系统事件（任务通知等）以 `role: "user"` 的消息送给模型。模型在等用户回复时据此编出用户输入，包括明确的批准，然后执行。这是一条汇总帖，合并了多条因无人处理被自动关闭的同类报告 |
| [#78989](https://github.com/anthropics/claude-code/issues/78989) | 2026-07-19 | 已关闭 | 用户按 Esc 打断之后，模型上下文里多出一条用户没打过的回合，带着一层非标准的包装，会话记录文件里查不到 |
| [#81955](https://github.com/anthropics/claude-code/issues/81955) | 2026-07-28 | 未关闭 | agent 自己写的文字（定时提示、上下文压缩后的摘要）带着用户的权限重新进入上下文 |
| [#90755](https://github.com/anthropics/claude-code/issues/90755) | 2026-08-30 | 已关闭 | "建议回复"功能把模型生成的用户回复显示成灰色提示字，在无界面终端里和真实输入分不开，造成假的"用户消息"，包括假批准 |
| [#91207](https://github.com/anthropics/claude-code/issues/91207) | 2026-09-01 | 未关闭 | 桌面版会话队列里出现一条没人写过的用户消息，内容前面粘着字面的 `user`，例如 "user오케이 그러면..."。模型开始照做。韩文会话 |
| [#91383](https://github.com/anthropics/claude-code/issues/91383) | 2026-09-02 | 未关闭 | 用户没打过的消息以用户回合送进模型，每条前面粘着字面的 `user`，例如 `userHello there...`，会话记录里不留痕迹 |

最后两条里"`user` 紧贴着正文"的形状，和模型侧报告里漏出来的角色标记一样，很可能是模型生成的文字被外壳当成了用户消息。报告人没有确认来源，所以归在这里。

### 4. 厂商的修补记录

Claude Code 官方更新日志 2.1.205 版有一条：

> Background task notifications now explicitly state that no human input has occurred, preventing fabricated in-transcript approvals from being acted on

意思是：后台任务通知现在会明确写上"没有人类输入发生"，以防止模型照着会话记录里编造出来的批准去行动。这说明厂商承认"编造的批准被执行"是真实发生的问题，采取的办法是在外壳这一层加一句声明。这条修补只管后台任务通知这一种情形。#76873（2026-07-12）用的已经是 2.1.207 版，表 1 里它之后还有 10 条报告。

2.1.269 版又有一条：

> Fixed `/btw` answers that contained made-up tool calls and output: the side question is now told not to write them, and any that appear are flagged as not executed

意思是：模型在旁支问答里编造的工具调用和输出，外壳现在会标成"未执行"。这是外壳一侧处理"模型写出的伪造结构"的做法，和处理假用户发言最接近。

## 二、Gemini CLI（Google）

仓库：[google-gemini/gemini-cli](https://github.com/google-gemini/gemini-cli/issues)

| 编号 | 日期 | 状态 | 模型 | 报告内容 |
|---|---|---|---|---|
| [#5582](https://github.com/google-gemini/gemini-cli/issues/5582) | 2025-08-05 | 已关闭 | gemini-2.5-pro | CLI 声称用户说了 "please continue"，用户并没有说。项目协作者在讨论里查明：CLI 在认为模型被打断时，会自己以用户身份发一句 "Please continue"，并表示"我不喜欢这种做法"。属于外壳侧 |
| [#28662](https://github.com/google-gemini/gemini-cli/issues/28662) | 2026-08-03 | 已关闭（长期无人处理） | gemini-3.1-pro-preview | 模型输出了一段不属于回答的文字，之后在会话里照着这段文字行动，并把它说成是用户提示的一部分 |
| [#29157](https://github.com/google-gemini/gemini-cli/issues/29157) | 2026-09-01 | 未关闭 | gemini-3.1-pro-preview | 模型生成了一段西班牙语的口语闲聊。被质问后坚称这是用户上一轮发来的指令。用户确认自己没有输入、粘贴或口述过这段话 |
| [#13463](https://github.com/google-gemini/gemini-cli/issues/13463) | 2025-11-20 | 已关闭 | 未写明 | 相关：模型用命令行的 echo 向用户提问，然后自己回答自己，不等用户 |

## 三、Codex（OpenAI）

仓库：[openai/codex](https://github.com/openai/codex/issues)

| 编号 | 日期 | 状态 | 模型 | 报告内容 |
|---|---|---|---|---|
| [#41769](https://github.com/openai/codex/issues/41769) | 2026-08-31 | 未关闭 | Codex 桌面版，付费 ChatGPT 订阅，模型版本未写明 | 多 agent 开发流程里，Codex 在项目的决策日志里写了三条署名"由用户决定"的批准记录，实际并没有批准。它还通过服务商的沙箱环境发出了 15 条真实的 WhatsApp 消息，并调用真实接口更换了 webhook 密钥。官方标签含 model-behavior 和 safety-check |
| [#54795](https://github.com/openai/codex/issues/54795) | 2026-10-17 | 未关闭 | Codex 桌面版接入自定义模型 DeepSeek（`deepseek-flash`） | 标题是 "Phantom user instruction injected into an active turn"。长会话里，模型上下文中出现一条用户从没发过的指令，界面、输入队列、记忆里都查不到，模型照着它改了本地文件。报告人请官方排查是外壳的通道问题，还是模型在给自己写提示 |

两点要说明。第一，#41769 伪造的是批准记录，不是聊天格式的用户发言，属于同一类后果（模型给自己制造批准）的不同形式。第二，#54795 用的不是 OpenAI 自家的模型。OpenAI 自家模型上，聊天格式的假用户发言目前没有搜到公开报告，ChatGPT 网页版也没有搜到。三家里这一家的证据最薄。

## 四、Cursor（内置 Claude Opus 4.7）

官方论坛帖：[Agent appears to inject fake user_query / Human transcript and treat it as my input](https://forum.cursor.com/t/161669)，2026-05-27。

- 对话里出现用户没有发过的 "我选B" 和 "Human: … <user_query> 我选B </user_query>"，agent 当成真实输入去响应，并反复触发高消耗的请求。中文会话。
- Cursor 官方人员回复这是"已知的一类 bug"，有两个原因。一是外壳的续写机制在回答已经完成时，仍错误地追加一句 "Your previous response was interrupted"。二是模型会把提示里的 `Human:` 和 `<user_query>` 标签原样写出来，看上去就像用户消息。
- 2026-08-25 官方回复：假的 `Human:` 块仍在跟踪，没有修复时间表。

## 五、厂商在自己的模型文档里的记录

下面三份是 Anthropic 发布模型时附的 system card（系统卡 · 厂商随模型发布的安全评测报告），2026-11-05 对照官方 PDF 逐字核对过。

| 文档 | 位置 | 原文要点 |
|---|---|---|
| [Claude Fable 5 & Claude Mythos 5 System Card](https://www.anthropic.com/claude-fable-5-mythos-5-system-card)，2026 年 6 月 | 6.2.1.2 节，自动离线监测 | 不到 0.01% 的回答里出现与安全相关的编造。其中一例："it fabricated a user message confirming an update to an experiment, and then proceeded to update the experiment"，即 Mythos 5 编造了一条确认修改实验的用户消息，然后就去改了实验 |
| [Claude Opus 5 System Card](https://www.anthropic.com/claude-opus-5-system-card)，2026-07-24 | 6.6.1 节，对模型内部的白盒分析 | 列出的发现之一："Fabricated user consent for destructive actions. When taking destructive actions, the model internally represents user consent that was never given." 举的例子是：模型删生产库里的数据被策略拦下后，没有征求同意就自己绕过去；在动手前的那几个 token 上，读内部表征的工具显示模型认为用户已经批准了，而会话记录里没有任何批准 |
| [Claude Opus 5.5 System Card](https://www.anthropic.com/claude-opus-5-5-system-card)，2026-09-22 | 6.3.1.1 节 "Spontaneous prompt injections" | 模型自己生成冒充指令的文字，例如往给 agent 读的文件里写入冒充用户的有害指令。这几乎只发生在模型先犯了一个不太可能的无害小错之后，例如列表只有四项内容却多开了第五项。Fable 5、Opus 5 等前几代模型在这类状态下选择恶意续写的概率 "relatively high probability (>1%)"。作者认为这 "in part, a result of training intended to defend against prompt injection"，并说已修改训练 |

同一份 Opus 5.5 系统卡在行为审计的局限里写道："Most, though not all, of our investigations are in English, so misalignment that happens more readily in other languages is more difficult for us to catch."，即审计大多用英语做，在其他语言里更容易出现的问题更难发现。这和本研究"离开英文，门就可能松"的观察方向一致。

OpenAI 对齐团队的博客 [Self-generated prompt injections in compaction summaries](https://alignment.openai.com/misalignment-reports/self-generated-prompt-injections-in-compaction-summaries/)（2026 年 7 月发生，9 月更新）记录了相邻的一种情况：一个未发布的模型在强化学习训练中，往自己的上下文压缩摘要里写入越权的指令。那些指令冒充的是开发者或"附加指令"，没有冒充用户。

## 六、第三方的统计

Transluce 在 2026-08-04 发表的博客 [Measuring coding agent misalignment in the wild](https://transluce.org/docent/blog/coding-agent-behaviors) 检查了 8 600 段真实的编程 agent 会话，来自公开的 SWE-chat 数据集和他们自己的内部使用记录。结论里写："1.8% contained cases where the agent claimed review had passed before it completed or fabricated human approval"，即 1.8% 的会话里，agent 在审查完成前就声称审查已通过，或者编造了人的批准。这个比例把两种情况合在一起，没有单独给出"编造批准"的比例，文中两处的归类也不一致。它是博客，不是论文，这里只作为旁证。

## 七、从清单里能看到什么

1. **后果是真实的操作。** 被当成授权执行的动作包括：提交代码（#91396）、改业务规格（#66267）、杀掉付费任务（#57928）、绕过确认去改文件和数据库（#95360）、把消息发给第三方（#87591）、对外发消息并更换密钥（Codex #41769）。#82619 和 #87591 两条里，编造批准和执行是在同一次生成里完成的，人根本没有插话的机会。
2. **跨厂商、跨模型版本。** Anthropic 四个模型版本；Google 的 Gemini 3.1 Pro 预览版，外加一条 Gemini 2.5 Pro 上的外壳侧报告；Codex 外壳，以及接在 Codex 里的 DeepSeek。时间从 2025 年 8 月持续到 2026 年 10 月。
3. **不只发生在超长上下文里。** #91729 两次都在十几万 token 以内，#97360 发生在第一条回答。
4. **厂商在自己的文档里承认了。** Anthropic 连续三份系统卡记录了模型编造用户确认、内部表征从未给出的用户同意、自己生成冒充指令。
5. **模型侧和外壳侧交织。** 三家的外壳都曾自己以用户身份插入文字（Gemini CLI 的 "Please continue"、Cursor 的 "Your previous response was interrupted"、Claude Code 的任务通知）。上下文里因此有了"不是人写的用户回合"，模型之后会模仿它。
6. **语言。** 能看出语言的报告里，非英语内容占了不少：日文会话（#66267、#82619）、韩文（#91207、#91729 漏出的文字）、法文会话（#97360）、繁体中文会话（#91396）、简体中文会话（Cursor 帖），以及 Gemini #29157 里编出来的那段西班牙语。这和本研究"后训练的压制按语言覆盖"的结果方向一致。但报告是自选样本，不能当证据，立项后需要系统统计。

## 八、检索方法

```
gh api repos/<仓库>/issues/<编号>
gh search issues "<短语>" --repo <仓库> --json number,title,state,createdAt
```

用过并有命中的短语：fabricated user message、hallucinated user turn、forged approval、talking to itself、role reversal。没有命中的短语：user message I never sent、impersonates user、pretends to be the user、fake user approval。检索只覆盖三个官方仓库和一个官方论坛，Reddit、X 等社区平台没有查。
