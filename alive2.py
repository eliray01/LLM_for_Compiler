import os
import subprocess
import tempfile



def compare_llvm_ir_with_alive2(src_ir: str, tgt_ir: str, alive_tv_path: str, timeout: int = 15) -> dict:
    """
    Compare two LLVM IR codes using Alive2's alive-tv.

    Args:
        src_ir (str): Source LLVM IR (e.g., model output).
        tgt_ir (str): Target LLVM IR (e.g., ground truth).
        alive_tv_path (str): Path to the alive-tv executable.
        timeout (int): Timeout in seconds for the subprocess.

    Returns:
        dict: A dictionary containing 'status', 'stdout', and 'stderr'.
    """
    with tempfile.TemporaryDirectory() as temp_dir:
        src_file = os.path.join(temp_dir, "src.ll")
        tgt_file = os.path.join(temp_dir, "tgt.ll")

        with open(src_file, 'w') as f:
            f.write(src_ir)
        with open(tgt_file, 'w') as f:
            f.write(tgt_ir)

        try:
            process = subprocess.Popen(
                [alive_tv_path, src_file, tgt_file],
                stdout=subprocess.PIPE,
                stderr=subprocess.PIPE,
                text=True
            )
            stdout, stderr = process.communicate(timeout=timeout)
            status = "success" if process.returncode == 0 else "failure"
        except subprocess.TimeoutExpired:
            process.kill()
            stdout, stderr = "", "Alive2 comparison timed out."
            status = "timeout"
        except Exception as e:
            stdout, stderr = "", str(e)
            status = "error"

        return {
            "status": status,
            "stdout": stdout,
            "stderr": stderr
        }
