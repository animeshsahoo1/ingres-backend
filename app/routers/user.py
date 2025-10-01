from fastapi import APIRouter,HTTPException,status, Query

userRouter = APIRouter()


@userRouter.get("/ping")
def pong():
    return {"ping": "pong!"}


@userRouter.post("/chat", status_code=status.HTTP_200_OK)
def user_chat(userMessage: str=Query(..., min_length=1, max_length=500)):
    if not userMessage or userMessage.isspace():
        raise HTTPException(status_code=400, detail="userMessage cannot be empty")
    return {"message": userMessage}