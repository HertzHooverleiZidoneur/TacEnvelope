# TacEnvelope 发布说明

当前目标是个人仓库 [HertzHooverleiZidoneur/TacEnvelope](https://github.com/HertzHooverleiZidoneur/TacEnvelope)。实验室官网暂不修改。

## 内容与预览

根目录包含代码、配置、测试、四个原始模型文件和校准 JSON。`docs/` 包含英文主页、你提供的论文 PDF、完整视频、方法图、真实照片、交互式结果和引用信息。原始材料目录未改动。

直接打开 `docs/index.html`，或在本目录运行 `python -m http.server 8000 --directory docs` 后访问 http://localhost:8000。

## 首次上传

先在个人账号创建公开的 `TacEnvelope` 空仓库，不预先生成 README、许可证或 .gitignore。需要用有写入权限的账号登录：

```powershell
git credential-manager github login --username HertzHooverleiZidoneur --force
```

完成浏览器里的登录后，在本目录运行：

```powershell
.\scripts\publish_github.ps1
```

脚本验证目标仓库、提交并推送、开启 Pages，并触发部署。凭证只在进程内使用，不写入文件。脚本不会创建仓库、改变现有可见性或强制覆盖远程历史。如果远程已有独立提交，会停止以便先检查和合并。

手动配置方式：推送后，在仓库 Settings → Pages → Source 中选择 GitHub Actions，再运行 Deploy project page 工作流。

默认项目页地址为 https://HertzHooverleiZidoneur.github.io/TacEnvelope/，以部署成功后 GitHub 返回的实际地址为准。canonical 和分享元数据已配置为此默认地址。

## 后续接入实验室官网

实验室官网已有论文条目，位于官网仓库的 `research/index.md`。个人页面上线确认后，可给它添加：

```markdown
[Project](https://HertzHooverleiZidoneur.github.io/TacEnvelope/) · [Code](https://github.com/HertzHooverleiZidoneur/TacEnvelope) · [Video](https://HertzHooverleiZidoneur.github.io/TacEnvelope/#video)
```

补丁位于 `website-integration/lab-website.patch`，当前尚未应用。后续若转移到组织仓库，需要同步更新代码链接、canonical、分享图 URL、CITATION.cff 和发布脚本。

参考：[GitHub Pages 自定义工作流](https://docs.github.com/en/pages/getting-started-with-github-pages/using-custom-workflows-with-github-pages)。

## 本地验证记录（2026-11-05）

- 50 项独立 Python 测试通过；仅有 pytest 缓存目录写入权限警告，不影响测试结果。
- `tests/test_online_rl.py` 需要额外的机器人端 `rlt_openpi` 运行环境，未纳入这次本地独立测试和默认 CI。
- 页面 JavaScript 语法、YAML、22 个本地资源链接、锚点、图片替代文本已检查。
- 发布目录中的论文与视频 SHA-256 与你提供的原文件完全一致。
- 修正了一处原有测试对 Windows 路径分隔符的断言，算法源文件保持原样。
- 浏览器自动化因无法可靠识别当前网址而停止，因此未完成页面的浏览器视觉实测。
- 当前版本仅在本地准备，尚未推送或修改实验室官网。
