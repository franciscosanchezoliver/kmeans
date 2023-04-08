# GitHub

You can create your own CI/DI pipelines using GitHub. 
You'll find information about how to handle pipelines.

## Trigger branches
> on:\
>   push:\
>     branches:    \
>       - '*'         # matches every branch that doesn't contain a '/'\
>       - '*/*'       # matches every branch containing a single '/'\
>       - '**'        # matches every branch\
>       - '!master'   # excludes master

