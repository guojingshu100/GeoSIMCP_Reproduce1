# 1. 环境恢复结果

Rscript 路径：`C:/Program Files/R/R-4.6.1/bin/Rscript.exe`。版本 4.6.1（2026-06-24 ucrt），平台 x86_64-w64-mingw32，architecture x86_64。绝对路径及 temporary script 调用成功；PATH 中仍未发现 Rscript，没有修改全局 PATH。

`R.home()` 返回短路径 `C:/PROGRA~1/R/R-46~1.1`，对应上述目录。完整 R.version、sessionInfo、capabilities 见 `r_environment_v1.log`。

## 2. Locale 与 library paths

首次启动报告 LC_COLLATE/LC_CTYPE/LC_MONETARY/LC_TIME 的 C.UTF-8 设置失败，实际回退 C。后续仅在 R 子进程设置 `LC_ALL=C`，不修改系统 locale；后续没有这些 warning。

`.libPaths()`：

1. `C:/Users/12068/AppData/Local/R/win-library/4.6`
2. `C:/Program Files/R/R-4.6.1/library`

## 3. 必需包与安装

历史 PM2.5 runner 加载 parallel、foreach、doParallel。GCCM4Lattice.r 本身没有额外 library/require/pkg::/source；使用 R base/stats 及并行分支函数。未引入 jsonlite、sf 等其他包。

| 包 | 版本 | 处理 |
|---|---|---|
| foreach | 1.5.2 | 本轮安装，用户 library |
| doParallel | 1.0.17 | 本轮安装，用户 library |
| iterators | 1.0.14 | 必需依赖，本轮安装 |
| codetools | 0.2-20 | 已有，未重装 |
| parallel | 4.6.1 | 已有，未重装 |

源：`https://cloud.r-project.org`。命令：绝对路径 Rscript `--vanilla install_required_packages_v1.R`。dependencies=NA，只安装必要 Depends/Imports/LinkingTo，不装 Suggests。用户本轮授权安装，写用户 library 的沙箱权限亦已获得。Return code=0；包 MD5 检查及 requireNamespace 成功。详见 `package_install_v1.log`，包括路径、版本、repository 和依赖。

PowerShell 将 R informational stderr 包装为 NativeCommandError 文本，但实际 Rscript return code=0、包验证通过；这不是安装失败。

## 4. Smoke tests

64 个临时 synthetic points，Delaunay、node ordering、Ring1–3、64×3 finite embedding、独立 Python BFS 数值比较、simplex、library loop、两次 hashes 全部通过。日志为 smoke_a.log、smoke_b.log，结构记录为 smoke_check_v1.json。

原始环境阻断记录已备份到 backup_20260909_101053。
