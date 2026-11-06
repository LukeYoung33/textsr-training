"""Create the Chinese offline data catalog from collected metadata."""
import argparse
from collections import Counter
import html
import json
from pathlib import Path


def esc(value):
    """@luna-comment Input:any display value. Output:escaped HTML text.
    Key Steps:string conversion and escaping. Key Params:no raw source HTML.
    State/Dependencies:pure formatting helper.
    """
    return html.escape(str(value))


def paths(linux, windows=''):
    """@luna-comment Input:actual Linux and mapped Windows paths. Output:copyable path markup.
    Key Steps:escape and label both paths. Key Params:blank Windows stays remote-only.
    State/Dependencies:pure formatting, no network requests.
    """
    return '<code>'+esc(windows or '仅 Linux：'+linux)+'</code>'+('<small>'+esc(linux)+'</small>' if windows else '')


def main():
    """@luna-comment Input:complete collected catalog directory. Output:offline Chinese HTML
    and English technical README. Key Steps:render sources, exact model bindings,
    counts, warning evidence and metadata download links; retain duplicate-view caveats.
    Key Params:no image data copied or unique training-image totals fabricated.
    State/Dependencies:own output directory only; browser uses embedded filter script.
    """
    parser=argparse.ArgumentParser();parser.add_argument('directory',type=Path);args=parser.parse_args();root=args.directory
    load=lambda name:json.loads((root/name).read_text(encoding='utf-8'))
    datasets=load('datasets.json');models=load('models.json');report=load('collection.json');raw=load('raw_roots.json');warnings=load('warnings.json')
    names={'v3_initial_20260820':'V3 Haar 初始化版本','0921_original':'0921 original','0928_main':'0928 主训练','0928_finetune':'0928 Mojito 微调','v5_four_20261014':'1014 V5 四组','v3_native_20261015':'1015 V3 原尺寸四组','fullframe_20261014':'1014 整图 C1/C2'}
    stages={'full_images':'文字区域大图（已裁剪）','text_lines':'文本行','text_lines_preprocessed':'固化预处理文本行（仅 LQ）','text_lines_ppocrv6_medium':'补充 OCR 文本行','native_patch_index':'原尺寸 patch 坐标索引','runtime_random_fullframe_patches':'整图运行时随机 patch','valid_cropped_parent_images':'0921 有效区域大图','raw_source_pairs':'原始图像配对关系','training_manifest':'训练使用清单','source_manifest':'原始来源清单','source_manifest_original':'历史原始来源清单','derived_source_manifest':'派生来源清单'}
    policies={'offline paired input':'离线配对输入','online_GT_degradation':'GT 在线退化','paired100%':'真实配对 100%','paired75% + online_GT_degradation25%':'真实配对 75% / GT 在线退化 25%'}
    stages['selection_decisions_metadata']='样本筛选审计记录（非训练图片）'
    modelnames={m['id']:m['name'] for m in models}
    pieces=['''<!doctype html><html lang="zh-CN"><meta charset="utf-8"><meta name="viewport" content="width=device-width,initial-scale=1"><title>训练数据完整索引</title><style>
    *{box-sizing:border-box}body{margin:0;background:#f3f5f7;color:#182333;font:15px/1.65 "Microsoft YaHei",sans-serif}main{max-width:1600px;margin:auto;padding:28px}h1{font-size:30px;margin:0}h2{margin-top:32px;font-size:22px}p{max-width:1200px}.muted,small{color:#617184}small{display:block;font-size:12px;overflow-wrap:anywhere}code{font:13px/1.6 Consolas,monospace;overflow-wrap:anywhere;user-select:all}a{color:#1461b3}.cards{display:grid;grid-template-columns:repeat(auto-fit,minmax(250px,1fr));gap:12px}.card,.note{background:white;border:1px solid #dce3eb;border-radius:10px;padding:16px}.card strong{font-size:26px;display:block}.note{border-left:4px solid #3382bb}.warn{border-left:4px solid #c27816;background:#fff9ed}table{border-collapse:collapse;width:100%;background:white}th,td{text-align:left;padding:12px;border:1px solid #dce3eb;vertical-align:top}th{background:#e8eef5;position:sticky;top:0}td.path{min-width:300px;max-width:650px}.scroll{overflow:auto}input,select{padding:10px;border:1px solid #bccada;border-radius:6px;margin:5px 8px 14px 0;min-width:260px}details{margin:6px 0}summary{cursor:pointer;color:#1461b3}.tag{display:inline-block;background:#edf3fa;padding:1px 6px;border-radius:4px}nav{display:flex;gap:18px;flex-wrap:wrap;margin:16px 0}.downloads{display:flex;gap:16px;flex-wrap:wrap}ul{padding-left:22px}td ul{margin:0}button{cursor:pointer}tr[hidden]{display:none}.number{white-space:nowrap}</style><main>
    <h1>训练数据完整索引</h1><p class="muted">2026-10-20 · 沿用现有数据目录 · 只整理元数据与清单，不复制训练图片</p>
    <nav><a href="#primary">主要数据目录</a><a href="#raw">原始大图</a><a href="#models">模型与数据对应</a><a href="#datasets">完整划分清单</a><a href="#checks">核对结果</a><a href="#downloads">下载清单</a></nav>''']
    pieces.append('<div class="cards">'+''.join(f'<div class="card">{label}<strong>{value:,}</strong></div>' for label,value in [('训练版本',report.get('unique_training_runs',len(models))),('数据集 / 划分视图',len(datasets)),('原尺寸 patch 坐标行',report['native_patch_rows']),('原始目录文件',sum(r['files'] for r in raw))])+'</div>')
    pieces.append(f'<p class="muted">保留 {len(models)} 份原始训练配置。同一微调目录中的历史配置分别列出，避免丢失阶段差异；并非 {len(models)} 个独立训练模型。</p>')
    pieces.append('''<div class="note" style="margin-top:18px"><b>阅读说明</b><ul><li><code>full_images</code> 是发布后的文字区域大图，部分经过有效区域裁剪；真正的相机原图请查看下方“原始大图”。</li><li>1014 与 1015 共享 5,542 个大图父样本。1015 主要更新 OCR / 文本行覆盖，不能把两个版本相加当作独立拍摄数据。</li><li>原尺寸 V3 使用大图 + patch 坐标在训练时裁出 96×480 输入。整图 C1/C2 使用运行时 256×256 随机裁剪；两者均没有预先保存全部 patch PNG。</li><li>“GT 在线退化”在训练时生成 LQ，不存在一套可交付的静态退化图片；精确退化配置保存在各模型的 YAML 快照中。</li><li>同一目录可能被多个模型 / 清单复用，所有表格记录的是视图与来源关系，不能简单累加样本行数作为唯一图像数量。</li></ul></div>''')
    pieces.append('<h2 id="primary">主要数据目录</h2><div class="scroll"><table><tr><th>用途</th><th>现有路径</th><th>说明</th></tr>')
    primary=[('1014 原始发布数据',r'Y:\guang.yang\TextSR\data\displayfit_training_20261014','full_images / text_lines；5,542 个父样本，原文本行 70,058 条。'),('1014 固化预处理 LQ',r'Y:\guang.yang\TextSR\data\displayfit_training_20261014\text_lines_preprocessed','frozen_strict_combined_v1；只修改 LQ，GT / OCR / mask 沿用。'),('1015 更新数据',r'Y:\guang.yang\TextSR\data\displayfit_training_20261015','同批父样本；文本行 77,696 条：train 68,188 / val 8,177 / review_excluded 1,331。'),('V3 四组原尺寸 patch 清单',r'Y:\guang.yang\TextSR\Lamu\N2N\experiments\v3_native_four_20261015','old_train / old_val / new_train / new_val.jsonl；包含源图、坐标和哈希。'),('0921 有效区域数据','/home/guang.yang/data/lenovo_textsr_20260920_final_validcrop','Linux 专用路径；不虚构对应 Y 盘路径。'),('整图 C1/C2 数据变体',r'Y:\guang.yang\TextSR\data\displayfit_training_20261014\full_images','GT、GT_VOSR、LQ_basic_block512_g1p8、GT_VOSR_basic_block512_g1p8；按每个配置清单选取。')]
    for label,path,description in primary:pieces.append(f'<tr><td>{label}</td><td class="path"><code>{esc(path)}</code></td><td>{description}</td></tr>')
    pieces.append('</table></div><h2 id="raw">原始大图与参考图</h2><p>下表已枚举目录中的全部文件。<a href="raw_files.csv.gz">完整原始文件清单</a>；逐样本“原图 → 对齐 → 裁剪”关系见 <a href="upstream_lineage.jsonl.gz">上游来源清单</a>。目录存在不代表其中每张图都参与了训练，训练成员以对应 manifest 为准。</p><div class="scroll"><table><tr><th>来源 / 角色</th><th>目录</th><th>文件 / 图像</th><th>大小</th></tr>')
    for row in raw:pieces.append(f'<tr><td>{esc(row["family"])}<br>{esc(row["role"])}</td><td class="path">{paths(row["linux_path"],row["windows_path"])}</td><td>{row["files"]:,} / {row["image_files"]:,}<small>{"已核对" if row["exists"] else "目录缺失"}; 错误 {len(row["errors"])}</small></td><td>{row["bytes"]/1024**3:.2f} GiB</td></tr>')
    pieces.append('</table></div><h2 id="models">模型与训练数据对应</h2><input id="modelSearch" placeholder="搜索：0921、0928、raw_h96、degraded、C1…"><div class="scroll"><table id="modelTable"><thead><tr><th>模型 / 训练轮次</th><th>实际数据 / 策略</th><th>配置与训练产物</th></tr></thead><tbody>')
    byid={d['id']:d for d in datasets}
    for m in models:
        bindings=[]
        for ident in m['dataset_ids']:
            d=byid[ident];bindings.append(f'<li><a href="#dataset-{ident}">{ident}</a> · {esc(d["split"])} · {d["samples"]:,} 行'+(f' / {d["patches"]:,} patches' if d['patches'] else '')+'</li>')
        preprocess=m.get('runtime_lq_preprocess') or {}
        bindings.append('<li>训练时 LQ 预处理：'+(esc(preprocess.get('method','已启用')) if preprocess.get('enabled') else '关闭（离线预处理以源数据为准）')+'</li>')
        bindings.append('<li>配置版本：<code>'+esc(Path(m['config_linux']).name)+'</code></li>')
        weights=m['observed_generator_files'];weight_detail=''
        if weights:weight_detail='<details><summary>查看已保存权重路径</summary>'+''.join('<code>'+esc(p)+'</code><br>' for p in weights)+'</details>'
        pieces.append(f'<tr><td><span class="tag">{m["id"]}</span> {esc(names.get(m["family"],m["family"]))}<br><b>{esc(m["name"])}</b><small>{esc(m.get("network_type"))}</small></td><td>{esc(policies.get(m["input_policy"],m["input_policy"]))}<ul>{"".join(bindings)}</ul></td><td><a href="{esc(m["config_snapshot"])}">原始 YAML 快照</a><small>SHA256 {esc(m["config_sha256"])}</small><details><summary>配置来源路径</summary>{paths(m["config_linux"],m["config_windows"])}</details>{"已找到权重文件" if weights else "此路径未定位到权重（不据此判定未训练）"}{weight_detail}</td></tr>')
    pieces.append('</tbody></table></div><h2 id="datasets">完整数据与划分</h2><input id="datasetSearch" placeholder="搜索数据来源、split、模型、路径…"><select id="splitFilter"><option value="">全部划分</option>'+''.join(f'<option>{esc(s)}</option>' for s in sorted({d['split'] for d in datasets}))+'</select><p class="muted">每条 manifest 均保存完整 gzip 快照与 SHA256；样本对应文件见 <a href="samples.csv.gz">完整样本表</a>。</p><div class="scroll"><table id="datasetTable"><thead><tr><th>编号 / 阶段</th><th>目录与源清单</th><th>划分 / 数量</th><th>对应模型</th></tr></thead><tbody>')
    for d in datasets:
        snapshot=f'<a href="{esc(d["snapshot"])}">完整 manifest 快照</a>' if d.get('snapshot') else '<b>源 manifest 缺失</b>'
        pieces.append(f'<tr id="dataset-{d["id"]}" data-split="{esc(d["split"])}"><td><b>{d["id"]}</b><br>{esc(stages.get(d["stage"],d["stage"]))}<small>{esc(d["family"])}</small></td><td class="path">{paths(d["root_linux"],d["root_windows"])}<details><summary>manifest 路径 / SHA256</summary>{paths(d["manifest_linux"],d["manifest_windows"])}<small>{esc(d.get("manifest_sha256","未采集"))}</small></details>{snapshot}</td><td class="number">{esc(d["split"])}<br>{d["samples"]:,} 行'+(f'<br>{d["patches"]:,} patches' if d['patches'] else '')+f'<small>缺失主文件引用：{d.get("core_missing_references",0):,}</small></td><td>{"<br>".join(esc(i+" · "+modelnames[i]) for i in d["used_by"]) or "来源 / 发布视图"}</td></tr>')
    pieces.append('</tbody></table></div><h2 id="checks">核对结果与边界</h2>')
    pieces.append(f'<div class="note {"warn" if report["missing_core_files"] else ""}">已检查 {report["unique_declared_files"]:,} 个去重文件路径，其中存在 {report["existing_files"]:,} 个；缺失 {report["missing_files"]:,} 个，其中主训练文件 {report["missing_core_files"]:,} 个。完整逐项证据见 <a href="missing_files.csv">缺失路径清单</a>。来源字段中的历史中间路径也单独保留，不能把所有来源缺失等同于当前训练数据缺失。</div>')
    pieces.append(f'<p>另有 {report.get("unresolved_relative_provenance_rows",0):,} 条继承的相对来源引用，基准目录未明确声明，单独保存在 <a href="unresolved_relative_references.csv.gz">未解析来源引用清单</a>，不计入缺失文件。已归档 {report["lineage_rows"]:,} 条大图上游关系。检查范围：清单 / 配置 SHA256、可解析文件路径存在性、原始目录文件数量与大小；没有重新读取并计算全部图片内容哈希。样本中已有的图片哈希仅作为发布时证据保留。</p><details><summary>采集告警（{len(warnings)} 条）</summary><pre style="white-space:pre-wrap">'+esc(json.dumps(warnings,ensure_ascii=False,indent=2))+'</pre></details>')
    downloads=[('datasets.csv','数据目录与划分 CSV'),('models.csv','模型配置 CSV'),('model_dataset_bindings.csv','模型与数据绑定 CSV'),('samples.csv.gz','全部样本与文件路径 CSV.gz'),('native_patches.csv.gz','原尺寸 patch 坐标 CSV.gz'),('raw_files.csv.gz','全部原始文件 CSV.gz'),('raw_roots.json','原始目录统计 JSON'),('upstream_lineage.jsonl.gz','逐图上游来源 JSONL.gz'),('referenced_files.csv.gz','全部引用文件存在性 CSV.gz'),('missing_files.csv','缺失路径 CSV'),('unresolved_relative_references.csv.gz','未解析的相对来源引用 CSV.gz'),('metadata_sources.json','发布与预处理元数据来源'),('validation.json','清单校验结果 JSON'),('collection.json','采集统计 JSON'),('README.md','技术说明')]
    pieces.append('<h2 id="downloads">清单下载</h2><div class="downloads">'+''.join(f'<a href="{file}">{label}</a>' for file,label in downloads)+'</div>')
    pieces.append('''<p class="muted">CSV 汇总使用 UTF-8 BOM，可直接用 Excel 打开；较大的样本表与原始 manifest 经过 gzip 压缩。索引页面离线可用，训练图片继续保留在原目录。</p></main><script>
    function filter(table,input,split){const q=document.getElementById(input).value.toLowerCase();const s=split?document.getElementById(split).value:'';document.querySelectorAll('#'+table+' tbody tr').forEach(r=>r.hidden=!(r.textContent.toLowerCase().includes(q)&&(!s||r.dataset.split===s)));}
    document.getElementById('modelSearch').addEventListener('input',()=>filter('modelTable','modelSearch'));
    document.getElementById('datasetSearch').addEventListener('input',()=>filter('datasetTable','datasetSearch','splitFilter'));
    document.getElementById('splitFilter').addEventListener('change',()=>filter('datasetTable','datasetSearch','splitFilter'));
    </script></html>''')
    (root/'index.html').write_text(''.join(pieces),encoding='utf-8')
    (root/'README.md').write_text('''# Historical training dataset handoff

This is a metadata-only handoff collected on 2026-10-20. Existing source data
directories are retained; no training images are copied. Open index.html for the
Chinese offline catalog. Directory existence and declared file existence were
checked on Remote58. Manifest/config SHA256 values refer to exact source bytes.
Published image hashes are preserved but were not recomputed for all image files.

## Catalog files

- datasets.csv/json: dataset root, exact manifest, split, counts and model users.
- models.csv/json and model_dataset_bindings.csv/json: original model configurations,
  source membership and online/offline input policy. Config snapshots preserve raw YAML.
- samples.csv.gz: one row per original manifest record, normalized main asset paths,
  existing published hashes, source identity and supplementary declared paths.
  Identifier priority is sample_id/source_id/pair_id/id, then segment_id/stem/frame_id,
  with manifest row number as a last resort. Untouched snapshots retain all source fields.
- manifests/*.jsonl.gz: complete exact source manifest bytes compressed losslessly.
- native_patches.csv.gz: every explicit native patch XY coordinate, width480/height96.
- raw_roots.json and raw_files.csv.gz: bounded original camera/reference directory
  catalog; a directory inventory does not imply every image was used for training.
- upstream_lineage.jsonl.gz: published parent -> metadata -> aligned/raw source links.
  Only explicit provenance/pair mappings are joined; the raw directory inventory
  does not promise recovered raw-photo pairing for every historical record.
- referenced_files.csv.gz and missing_files.csv: deduplicated declared file existence;
  distinguish core training paths from historical provenance references.
- unresolved_relative_references.csv.gz: inherited relative provenance fields whose
  base directory is not declared. These are not classified as missing files; normalized
  supplementary assets retain only a clearly labeled unresolved candidate path.

## Interpretation

Do not sum sample rows across reused manifests, splits, release versions or model
bindings as unique images. Referenced files are deduplicated by declared absolute
path string; symlink, hardlink and lexical aliases are not merged by inode.
Displayfit1014/1015 share5542 cropped parent images;
full_images is a region-cropped release, not a guarantee of original camera dimensions.
Text-line membership differs between releases. review_excluded is not a training split.
Online degraded LQ is generated at training time, not a stored image collection.
Native V3 uses source images plus96x480 coordinate indexes; fullframe C1/C2 use
runtime random256x256 sampling. C1/C2 select different LQ/GT variants through their
own manifests even when they share one root. Initial V3 source manifests and0928
main/finetune source manifests are included separately from later release views.
Only /AI_DNS paths have an established Y drive mapping. /home and /data remain
Linux-only paths. Missing paths are retained as explicit evidence. A missing weight
lookup is not conclusive proof that a configured model was never trained.
Malformed historical YAML standalone trailing port fields may be ignored in-memory
for metadata parsing; original YAML bytes and the parsing warning are both preserved.
''',encoding='utf-8')
    print(json.dumps(dict(html=str(root/'index.html'),models=len(models),datasets=len(datasets)),ensure_ascii=True))


if __name__=='__main__':main()
