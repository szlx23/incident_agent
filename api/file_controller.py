import logging
from pathlib import Path

from fastapi import APIRouter, UploadFile, File, HTTPException
from starlette.responses import JSONResponse
from werkzeug.utils import secure_filename

from config import Config
from service.rag.index_service import IndexService

router = APIRouter()



@router.post("/upload")
async def upload_file(file: UploadFile = File(...)):
    # 第一步：校验并净化文件名，阻断空文件名、特殊字符和潜在目录穿越风险。
    if not file.filename:
        raise HTTPException(status_code=400, detail="文件名不能为空")

    safe_filename = secure_filename(file.filename)
    if not safe_filename:
        raise HTTPException(status_code=400, detail="非法文件名")

    if "." not in safe_filename:
        raise HTTPException(status_code=400, detail="文件必须包含拓展名")

    file_extension = safe_filename.rsplit(".", 1)[-1].lower()
    if file_extension not in Config.allow_extension:
        raise HTTPException(
            status_code=400,
            detail=f'不支持的文件格式, 仅支持 {", ".join(Config.allow_extension)}'
        )

    upload_dir = Path(Config.upload_dir)
    # 启动时未创建目录也能工作，按需创建上传目录。
    upload_dir.mkdir(parents=True, exist_ok=True)

    file_path = (upload_dir / safe_filename).resolve()
    # 第二道路径校验：保证最终落盘路径仍在 upload_dir 内。
    if upload_dir not in file_path.parents:
        raise HTTPException(status_code=400, detail="非法文件路径")

    if file_path.exists():
        logging.info(f"文件已存在, 将覆盖 {file_path}")
        file_path.unlink()

    content_size = 0
    try:
        with open(file_path, "wb") as f:
            # 分块读取上传流并写盘，避免大文件一次性读入内存。
            while True:
                chunk = await file.read(1024 * 1024)
                if not chunk:
                    break
                f.write(chunk)
                content_size += len(chunk)
            logging.info(f"文件上传成功: {file_path}")
    except Exception as e:
        # 存盘失败直接中断请求，避免后续索引基于不完整文件执行。
        logging.error(f"文件保存失败: {e}")
        raise HTTPException(status_code=500, detail="文件保存失败")

    # 实时执行索引：只有索引成功才返回 200，保证“上传成功”与“可检索”一致。
    index_service = IndexService()
    try:
        index_service.index_single_file(file_path=str(file_path))
        logging.info(f"向量索引创建成功: {file_path}")
    except Exception as e:
        logging.error(f"向量索引创建失败: {file_path}, 错误: {e}")
        raise HTTPException(status_code=500, detail="向量索引创建失败")

    return JSONResponse(
        status_code=200,
        content={
            "code": 200,
            "message": "success",
            "data": {
                "filename": safe_filename,
                "file_path": str(file_path),
                "size": content_size
            }
        }
    )
