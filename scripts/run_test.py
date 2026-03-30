import sys
import traceback
try:
    import train_llama
    train_llama.main()
except BaseException:
    with open("python_err.log", "w") as f:
        traceback.print_exc(file=f)
