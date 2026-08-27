#import "@preview/cuti:0.2.1": show-cn-fakebold

#show: show-cn-fakebold
#set text(font: ("Palatino Linotype", "kaiti"))
#set math.equation(numbering: "(1)")
#set page(numbering: "1")
#set heading(numbering: "1.1")
#show enum: it => {
  set block(spacing: 0.5em)
  pad(left: 2em, it)
}

#let centertitle(body) = [
  #set text(size: 24pt, weight: "bold")
  #align(center)[
    #v(1em)
    #body
    #v(0.5em)
  ]
]

#let info-row(name, value) = [
  #align(center)[#name：#value]
]

#let screenshot(path, caption) = figure(
  image(path, width: 85%),
  caption: caption,
)

#let codeblock(body) = block(
  fill: rgb("#f6f8fa"),
  inset: 8pt,
  radius: 2pt,
  width: 100%,
)[#body]

#centertitle[HPC Lab1 Report]

#set par(first-line-indent: (amount: 2em, all: true), spacing: 1em, leading: 0.8em)

#show outline.entry.where(level: 1): it => {
  v(1.2em, weak: true)
  strong(it)
}

#outline(
  title: none,
  indent: 1.5em,
)
#pagebreak()

= 实验目的
#v(0.5em)
本实验围绕高性能计算集群的基础搭建、共享文件系统配置、作业调度系统部署以及并行基准测试展开。

本文主要完成以下内容：
#v(0.3em)
1. 搭建并记录实验所需的节点环境；
2. 配置 OpenMPI、BLAS、HPL 等高性能计算软件栈；
3. 配置 NFS 共享文件系统，实现多节点文件共享；
4. 配置 Slurm 作业调度系统，实现批处理作业提交与运行；
5. 使用 HPL 基准测试评估集群浮点计算性能；
6. 对实验过程中遇到的问题进行记录和分析。

= 实验环境
== 节点概况
#v(0.5em)

本实验在 Windows 主机的 WSL2 Ubuntu 环境中使用 Docker 容器模拟 MiniCluster。WSL 主机提供容器运行环境，实际参与集群调度和通信的节点为 `node01` 至 `node04` 四个容器，容器镜像为 `wsl-cluster-lab:ubuntu24.04`。容器网络使用 Docker bridge 网络，网段为 `10.42.0.0/24`。

#align(center)[
  #figure(
    table(
      columns: (auto, auto, auto, auto, auto),
      align: center + horizon,
      stroke: none,

      table.hline(stroke: 1pt),
      table.header(
        [节点], [IP 地址], [角色], [CPU / 核数], [内存]
      ),
      table.hline(stroke: 0.5pt),

      [node01], [10.42.0.11], [管理节点 / Slurm 控制节点], [22 vCPU 可见；Slurm 配置中不作为计算节点], [WSL 共享内存 15 GiB],
      [node02], [10.42.0.12], [计算节点 / Slurm `debug` 分区], [容器内可见 22 vCPU；Slurm 配置 1 CPU], [WSL 共享内存 15 GiB；Slurm 配置 512 MB],
      [node03], [10.42.0.13], [计算节点 / Slurm `debug` 分区], [容器内可见 22 vCPU；Slurm 配置 1 CPU], [WSL 共享内存 15 GiB；Slurm 配置 512 MB],
      [node04], [10.42.0.14], [计算节点 / Slurm `debug` 分区], [容器内可见 22 vCPU；Slurm 配置 1 CPU], [WSL 共享内存 15 GiB；Slurm 配置 512 MB],

      table.hline(stroke: 1pt),
    ),
    caption: [集群节点基本信息],
  )
]

== 系统与网络配置
#v(0.5em)

各节点均运行 Ubuntu 24.04 LTS（Noble Numbat）。在 WSL 主机中通过 `docker ps` 可以看到四个容器节点均处于运行状态：

#figure(
  image("assets/lab1/image-1.png")
)

以 `node01` 为例，节点基础信息如下。容器主机名与节点名一致，IP 地址为 `10.42.0.11, 10.244.0.0, 10.244.0.1`，操作系统为 Ubuntu 24.04 LTS：

#figure(
  image("assets/lab1/image-2.png")
)

CPU 和内存信息由容器继承 WSL2 主机资源。`lscpu` 显示容器中可见 22 个逻辑 CPU，处理器型号为 Intel Core Ultra 7 155H；`free -h` 显示可用内存约 15 GiB，交换空间 4.0 GiB：

#figure(
  image("assets/lab1/image-4.png")
)

主机名解析在容器 `/etc/hosts` 中静态配置，`node01` 到 `node04` 分别映射到 `10.42.0.11` 到 `10.42.0.14`：

#figure(
  image("assets/lab1/image-5.png", width: 40%)
)

网络连通性通过 `ping` 验证。从 `node01` 到三个计算节点均能正常通信。

#figure(
  image("assets/lab1/image-6.png", width: 60%)
)

Slurm 配置中 `node01` 为控制节点，`node02`、`node03`、`node04` 为计算节点。关键配置如下：

#figure(
  image("assets/lab1/image-7.png")
)

使用 `sinfo` 验证调度系统能够识别计算节点，`debug` 分区包含 3 个空闲节点：

#figure(
  image("assets/lab1/image-8.png", width: 60%)
)

各节点已启动 `sshd` 和 `munged`。`node01` 上运行 `slurmctld`，计算节点上运行 `slurmd`。

== SSH 免密与用户配置
#v(0.5em)

集群中所有节点使用相同的普通用户 `labuser`（UID=2001, GID=2001），保证 NFS 文件权限一致性。在 `node01` 上为 `labuser` 生成 ed25519 密钥对，并通过 `ssh-copy-id` 分发公钥到各计算节点，实现 `node01` 到 `node02`、`node03`、`node04` 的 SSH 免密登录：

#codeblock(```bash
ssh-keygen -t ed25519
ssh-copy-id labuser@node02
ssh-copy-id labuser@node03
ssh-copy-id labuser@node04
```)

各节点时间通过 WSL2 内核时钟统一同步，经 `date` 命令验证，节点间时间差在 1 秒以内，满足 MUNGE 认证要求。

= 软件栈安装与配置

== OpenMPI 安装
#v(0.5em)

本实验从源码编译安装 OpenMPI 4.1.6 到 `/opt/openmpi`，而非使用 apt 包管理器。在 WSL 主机中下载源码后，执行 configure 和 make 完成编译：

#codeblock(```bash
cd /tmp/lab1-build
wget https://www.open-mpi.org/software/ompi/v4.1/downloads/openmpi-4.1.6.tar.gz
tar xzf openmpi-4.1.6.tar.gz && cd openmpi-4.1.6
./configure --prefix=/opt/openmpi --enable-mpi-cxx
make -j$(nproc)
sudo make install && sudo ldconfig
```)

安装完成后，使用 `ompi_info` 验证编译结果。`Prefix` 为 `/opt/openmpi`，`Configure command line` 包含 `--prefix=/opt/openmpi --enable-mpi-cxx`，说明是从源码编译的版本：

#codeblock(```text
$ /opt/openmpi/bin/ompi_info --version
Open MPI v4.1.6

$ /opt/openmpi/bin/ompi_info -all | grep -E 'Prefix:|Configure command'
                  Prefix: /opt/openmpi
  Configure command line: '--prefix=/opt/openmpi' '--enable-mpi-cxx'
```)

源码编译的 OpenMPI 安装在 `/opt/openmpi`，容器内使用 apt 安装的 OpenMPI 4.1.6（同版本）提供运行时支持。两者版本一致，`mpirun` 和 `mpicc` 均可正常使用。

== BLAS 数学库编译
#v(0.5em)

BLAS（Basic Linear Algebra Subprograms）是 HPL 进行矩阵乘法、向量运算和 LU 分解时依赖的基础线性代数接口。本实验从源码编译参考 BLAS 3.12.0，生成 `blas_LINUX.a` 静态库，供 HPL 链接使用。

从 netlib 下载 BLAS 源码并编译：

#codeblock(```bash
wget https://www.netlib.org/blas/blas-3.12.0.tgz
tar xzf blas-3.12.0.tgz && cd BLAS-3.12.0
```)

修改 `make.inc` 中的 Fortran 编译器为 `gfortran`，并添加 `-fallow-argument-mismatch` 以兼容新版 gfortran。执行 `make` 后，目录下生成 `blas_LINUX.a`（约 586 KiB）：

#codeblock(```make
FORTRAN = gfortran
OPTS = -O3 -fallow-argument-mismatch
LOADER = gfortran
ARCH = ar
ARCHFLAGS = cr
```)

#codeblock(```bash
$ ls -la blas_LINUX.a
-rw-r--r-- 1 cx cx 600080 May 29 19:09 blas_LINUX.a
```)

== CBLAS 编译
#v(0.5em)

CBLAS 是 BLAS 的 C 语言接口封装，HPL 编译时通过 `-DHPL_CALL_CBLAS` 宏指定调用 CBLAS 接口。从 netlib 下载 CBLAS 源码，修改 `Makefile.in` 指向已编译的 BLAS 静态库：

#codeblock(```bash
wget https://www.netlib.org/blas/blast-forum/cblas.tgz
tar xzf cblas.tgz && cd CBLAS
```)

`Makefile.in` 关键修改项：

#codeblock(```make
BLLIB = ../BLAS-3.12.0/blas_LINUX.a
CBLIB = ../lib/cblas_$(PLAT).a
CC = gcc
FC = gfortran
CFLAGS = -O3 -DADD_
FFLAGS = -O3 -fallow-argument-mismatch
```)

执行 `make` 后，`lib/` 目录下生成 `cblas_LINUX.a`（约 387 KiB）：

#codeblock(```bash
$ ls -la lib/cblas_LINUX.a
-rw-r--r-- 1 cx cx 396314 May 29 19:14 lib/cblas_LINUX.a
```)

HPL 编译时将 BLAS 和 CBLAS 静态库链接进 `xhpl` 二进制文件中，因此 `ldd` 不会显示任何 BLAS 动态库依赖，这是预期行为：

#codeblock(```bash
$ ldd /cluster/shared/hpl/xhpl | grep -Ei 'blas|lapack|openblas' || echo "No external BLAS library found in xhpl"
No external BLAS library found in xhpl
```)

#figure(
  image("assets/lab1/image-10.png")
)

== HPL 编译
#v(0.5em)

从 netlib 下载 HPL 2.3 源码，复制 `Make.Linux_PII_FBLAS` 为 `Make.Linux` 并修改关键变量。`MPdir` 指向系统 OpenMPI 路径，`LAlib` 指向前面编译的 BLAS 和 CBLAS 静态库，`CC` 和 `LINKER` 使用 `mpicc`：

#codeblock(```bash
wget https://www.netlib.org/benchmark/hpl/hpl-2.3.tar.gz
tar xzf hpl-2.3.tar.gz && cd hpl-2.3
cp setup/Make.Linux_PII_FBLAS Make.Linux
make arch=Linux
```)

`Make.Linux` 关键配置：

#codeblock(```make
ARCH = Linux
TOPdir = /home/cx/workspace/lab1/hpl-2.3
MPdir = /usr
MPlib = -L$(MPdir)/lib/x86_64-linux-gnu -lmpi
LAdir = /home/cx/workspace/lab1/BLAS-3.12.0
LAlib = /home/cx/workspace/lab1/CBLAS/lib/cblas_LINUX.a \
        /home/cx/workspace/lab1/BLAS-3.12.0/blas_LINUX.a -lgfortran -lm
HPL_OPTS = -DHPL_CALL_CBLAS
CC = mpicc
LINKER = mpicc
```)

编译完成后，`bin/Linux/` 目录下生成 `xhpl` 可执行文件。通过 `ldd` 验证链接情况：`xhpl` 动态链接 `libmpi.so.40` 和 `libgfortran.so.5`，BLAS/CBLAS 以静态库形式链接进二进制文件，因此不出现 BLAS 动态依赖。将 `xhpl` 复制到共享目录 `/cluster/shared/hpl/`，计算节点可通过同一路径访问：

#codeblock(```text
/cluster/shared/hpl/xhpl
/cluster/shared/hpl/HPL.dat
/cluster/shared/hpl/run-hpl.sbatch
/cluster/shared/hpl/hpl-4.out
/cluster/shared/hpl/hpl-4.err
```)

当前 HPL 运行使用 OpenMPI 和 Slurm 启动。作业脚本 `run-hpl.sbatch` 的内容如下：

#codeblock(```bash
#!/bin/bash
#SBATCH --job-name=hpl
#SBATCH --partition=debug
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --output=/cluster/shared/hpl/hpl-%j.out
#SBATCH --error=/cluster/shared/hpl/hpl-%j.err

set -euo pipefail
cd /cluster/shared/hpl
srun --mpi=pmix -N3 --ntasks=3 ./xhpl
```)

HPL 输入文件 `HPL.dat` 中，本次验证运行采用 3 个 MPI rank，对应 `P=1`、`Q=3` 的进程网格，问题规模和块大小分别为 `N=120`、`NB=32`：

#figure(
  image("assets/lab1/image-11.png", width: 80%)
)

使用 Slurm 提交 HPL 的命令为：

#codeblock(```bash
sbatch /cluster/shared/hpl/run-hpl.sbatch
squeue
cat /cluster/shared/hpl/hpl-4.out
```)

已有运行结果显示，作业成功生成输出文件，并通过 HPL 残差校验；最终性能行为：

#figure(
  image("assets/lab1/image-12.png")
)

= NFS 共享文件系统配置

== 服务端配置
#v(0.5em)

本实验使用 NFS 将管理节点 `node01` 上的共享目录导出给计算节点使用。共享目录统一挂载为 `/cluster/shared`，HPL 可执行文件、输入参数文件和 Slurm 作业脚本均放置在该目录下。这样 `node02`、`node03`、`node04` 在运行作业时可以通过相同路径访问文件，避免每个节点分别复制程序和配置文件。

在 `node01` 上安装 NFS 服务端软件包，并创建共享目录：

#codeblock(```bash
apt update
apt install -y nfs-kernel-server
mkdir -p /cluster/shared
chmod 777 /cluster/shared
```)

随后编辑 `/etc/exports`，将 `/cluster/shared` 导出给容器网络 `10.42.0.0/24`。实验环境为 Docker bridge 网络，因此使用较宽松的读写权限，便于各计算节点共同访问实验文件：

#codeblock(```text
/cluster/shared 10.42.0.0/24(rw,sync,no_subtree_check,no_root_squash)
```)

配置完成后，重新导出共享目录并检查 NFS 服务状态：

#figure(
  image("assets/lab1/image-13.png")
)

== 客户端挂载
#v(0.5em)

计算节点作为 NFS 客户端，需要安装 `nfs-common`，并将 `node01:/cluster/shared` 挂载到本地同名目录 `/cluster/shared`。为了保证 HPL 作业脚本中使用的路径在所有节点上一致，客户端挂载点与服务端共享目录保持相同。

容器环境中没有启动 `rpc.statd`，直接使用默认 NFS 挂载参数会提示需要远程锁服务。因此本实验在客户端挂载时加入 `nolock` 参数，使锁状态保存在本地，满足本实验共享文件访问与 HPL 程序读取的需求。在 `node02`、`node03`、`node04` 上分别执行的挂载命令如下：

#codeblock(```bash
mkdir -p /cluster/shared
mount -o nolock -t nfs node01:/cluster/shared /cluster/shared
df -h | grep /cluster/shared
mount | grep /cluster/shared
```)

实际验证时，先使用 `showmount -e node01` 确认服务端已经导出 `/cluster/shared`，再检查本地挂载表。`node02` 的输出如下：

#figure(
  image("assets/lab1/image-14.png")
)

`node03` 的挂载结果如下：

#figure(
  image("assets/lab1/image-15.png")
)

`node04` 的挂载结果如下：

#figure(
  image("assets/lab1/image-16.png")
)

三个计算节点的 `df -h` 均显示 `node01:/cluster/shared` 已挂载到 `/cluster/shared`，`mount` 输出中也可以看到挂载类型为 `nfs`，并且挂载选项中包含 `rw` 和 `nolock`，说明客户端挂载配置生效。

== 共享读写验证
#v(0.5em)

完成挂载后，需要验证共享目录是否真正实现跨节点读写。本实验采用“计算节点写入、管理节点读取、其他计算节点继续读取”的方式检查文件一致性。

首先在 `node02` 上向共享目录写入测试文件：

#codeblock(```bash
hostname
echo "nfs test from node02" > /cluster/shared/nfs-test.txt
cat /cluster/shared/nfs-test.txt
ls -l /cluster/shared/nfs-test.txt
```)

然后在 `node01` 上读取同一文件，并追加一行内容：

#codeblock(```bash
hostname
cat /cluster/shared/nfs-test.txt
echo "checked from node01" >> /cluster/shared/nfs-test.txt
cat /cluster/shared/nfs-test.txt
```)

最后在另一个计算节点 `node03` 上再次读取该文件，若能够看到 `node02` 写入和 `node01` 追加的内容，说明 NFS 共享目录已经可以被多节点共同访问：

#codeblock(```bash
hostname
cat /cluster/shared/nfs-test.txt
ls -lah /cluster/shared
```)

HPL 文件也放置在该共享目录中。通过在计算节点上检查 `/cluster/shared/hpl`，可以确认后续 Slurm 作业能够从统一路径访问 `xhpl`、`HPL.dat` 和作业脚本：

#codeblock(```bash
hostname
ls -lah /cluster/shared/hpl
test -x /cluster/shared/hpl/xhpl && echo "xhpl executable"
```)

#figure(
  image("assets/lab1/image-17.png", width: 80%)
)
#figure(
  image("assets/lab1/image-18.png", width: 80%)
)

= Slurm 作业调度系统配置

== MUNGE 配置
#v(0.5em)

Slurm 使用 MUNGE 进行节点间身份认证。集群中所有参与调度的节点必须使用同一个 `/etc/munge/munge.key`，并且该文件需要由 `munge` 用户持有，权限应限制为仅所有者可读。否则 `slurmctld` 与 `slurmd` 之间无法完成认证，计算节点会无法注册到控制节点。

本实验中在 `node01` 生成 MUNGE key 后，将其复制到 `node02`、`node03`、`node04`，并在各节点启动 `munged`。在 `node01` 上检查 key 文件和认证服务的结果如下：

#codeblock(```bash
ls -l /etc/munge/munge.key
service munge status
munge -n | unmunge
```)

#figure(
  image("assets/lab1/image-19.png", width: 80%)
)

在计算节点上同样检查 `munged` 运行状态。以 `node02` 为例，服务已经正常启动：

#codeblock(```text
node02
munged (pid 23) is running
```)

== Slurm 配置文件
#v(0.5em)

Slurm 的核心配置文件为 `/etc/slurm/slurm.conf`。本实验中 `node01` 作为控制节点运行 `slurmctld`，`node02`、`node03`、`node04` 作为计算节点运行 `slurmd`。由于容器环境用于功能验证而非真实性能隔离，每个计算节点在 Slurm 中配置为 `CPUs=1`、`RealMemory=512`，并统一加入默认分区 `debug`。

关键配置如下.其中，`SlurmctldHost` 指定控制节点，`NodeName` 定义计算节点名称、IP 地址、CPU 数和内存资源，`PartitionName` 定义可提交作业的分区。本实验的作业均提交到 `debug` 分区。

#figure(
  image("assets/lab1/image-20.png")
)

== 服务状态与节点验证
#v(0.5em)

配置完成后，在 `node01` 启动 `slurmctld`，在计算节点启动 `slurmd`。服务启动后，使用 `sinfo` 查看分区状态，并用 `scontrol show nodes` 查看计算节点详细信息。

控制节点和分区识别结果如下：

#codeblock(```bash
hostname
service slurmctld status
sinfo
scontrol show nodes | grep -E 'NodeName=|NodeAddr=|State=|CfgTRES='
```)

#figure(
  image("assets/lab1/image-21.png", width: 80%)
)

计算节点服务状态以 `node02` 为例：

#codeblock(```bash
hostname
service munge status
service slurmd status
```)

#figure(
  image("assets/lab1/image-22.png")
)

以上结果表明，Slurm 控制节点已经启动，`debug` 分区处于 `up` 状态，三个计算节点均为 `idle`，可以接受作业调度。

== 作业提交测试
#v(0.5em)

为了验证 Slurm 能够完成作业提交、调度和执行，本实验在共享目录 `/cluster/shared` 中创建了一个简单的批处理脚本。脚本申请 `debug` 分区中的 1 个节点和 1 个任务，输出文件写入共享目录，便于从控制节点查看。

测试脚本 `/cluster/shared/slurm-test.sbatch` 内容如下：

#codeblock(```bash
#!/bin/bash
#SBATCH --job-name=slurm-test
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/cluster/shared/slurm-test-%j.out

hostname
pwd
srun hostname
```)

#figure(
  image("assets/lab1/image-23.png")
)

输出中第一行 `Submitted batch job 3` 表示作业提交成功。随后 `squeue` 没有显示正在排队的作业，说明该测试作业已经快速完成。输出文件中两次出现 `node02`，第一处来自批处理脚本直接执行的 `hostname`，第二处来自 `srun hostname`，说明作业被 Slurm 调度到计算节点 `node02` 并正常运行。


= HPL 基准测试

== 作业脚本
#v(0.5em)

HPL（High Performance Linpack）通过求解稠密线性方程组来评估浮点计算性能。在集群验证阶段，`xhpl`、`HPL.dat` 和 Slurm 作业脚本均放在 NFS 共享目录中，计算节点通过同一路径访问程序和输入文件。

实际使用的 `/cluster/shared/hpl/run-hpl.sbatch` 内容如下：

#codeblock(```bash
#!/bin/bash
#SBATCH --job-name=hpl
#SBATCH --partition=debug
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --output=/cluster/shared/hpl/hpl-%j.out
#SBATCH --error=/cluster/shared/hpl/hpl-%j.err

set -euo pipefail
cd /cluster/shared/hpl
srun --mpi=pmix -N3 --ntasks=3 ./xhpl
```)

脚本申请 `debug` 分区中的 3 个节点，每个节点 1 个 MPI 任务，因此总 MPI rank 数为 3。`HPL.dat` 中的进程网格必须满足 `P * Q = 3`，本次验证运行设置为 `P=1`、`Q=3`。输出和错误日志通过 `%j` 使用 Slurm 作业号命名，便于多次运行时保留不同结果。

== HPL.dat 参数
#v(0.5em)

HPL 的主要输入参数位于 `HPL.dat`。

集群验证运行使用的是小规模参数，目的是确认 Slurm、MPI、共享文件系统和 HPL 残差校验链路能够跑通，而不是追求最高性能。关键参数如下：

#align(center)[
  #figure(
    table(
      columns: (auto, auto, auto),
      align: center + horizon,
      stroke: none,

      table.hline(stroke: 1pt),
      table.header([参数], [取值], [说明]),
      table.hline(stroke: 0.5pt),

      [`N`], [`120`], [问题规模，矩阵阶数],
      [`NB`], [`32`], [分块大小],
      [`P`], [`1`], [进程网格行数],
      [`Q`], [`3`], [进程网格列数],

      table.hline(stroke: 1pt),
    ),
    caption: [HPL.dat 关键参数],
  )
]

== 运行结果
#v(0.5em)

提交作业时不固定使用某一个历史输出文件名，而是先保存 `sbatch --parsable` 返回的作业号，再根据作业号读取对应的 `hpl-${jobid}.out`。

已有运行结果保存在 `/cluster/shared/hpl/hpl-4.out`，关键输出如下：

#codeblock(```text
T/V    : Wall time / encoded variant.
Gflops : Rate of execution for solving the linear system.
T/V                N    NB     P     Q               Time                 Gflops
WR00R2R2         120    32     1     3               0.01             1.2523e-01
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=   1.34012681e-02 ...... PASSED
End of Tests.
```)

结果中 `PASSED` 表示残差校验通过，说明线性方程组求解结果满足 HPL 的正确性阈值。`Gflops=1.2523e-01` 是本次小规模集群验证的浮点性能。由于 `N=120` 很小，运行时间只有约 `0.01 s`，该数值主要用于验证配置正确性，不能代表 WSL 主机的稳定峰值性能。

== 参数调优与性能分析
#v(0.5em)

调优记录中的基线测试使用默认 `HPL.dat`，默认问题规模只有 `N=29、30、34、35`，虽然 864 组残差校验全部通过，但运行时间接近 0，GFLOPS 结果不适合作为性能分析依据。因此后续将问题规模提高到 `N=1000、2000、4000`，并固定使用更有代表性的参数组合进行比较。

4 个 MPI rank 的参数扫描结果如下：

#align(center)[
  #figure(
    table(
      columns: (auto, auto, auto, auto, auto, auto, auto, auto),
      align: center + horizon,
      stroke: none,

      table.hline(stroke: 1pt),
      table.header([节点数], [总进程数], [`P`], [`Q`], [`N`], [`NB`], [性能], [校验]),
      table.hline(stroke: 0.5pt),

      [本地], [`4`], [`1`], [`4`], [`1000`], [`64`], [`8.15 GFLOPS`], [PASSED],
      [本地], [`4`], [`1`], [`4`], [`1000`], [`192`], [`8.95 GFLOPS`], [PASSED],
      [本地], [`4`], [`2`], [`2`], [`1000`], [`64`], [`15.68 GFLOPS`], [PASSED],
      [本地], [`4`], [`1`], [`4`], [`2000`], [`64`], [`18.91 GFLOPS`], [PASSED],
      [本地], [`4`], [`1`], [`4`], [`2000`], [`128`], [`15.09 GFLOPS`], [PASSED],
      [本地], [`4`], [`2`], [`2`], [`2000`], [`64`], [`17.62 GFLOPS`], [PASSED],
      [本地], [`4`], [`4`], [`1`], [`2000`], [`64`], [`15.64 GFLOPS`], [PASSED],

      table.hline(stroke: 1pt),
    ),
    caption: [4 rank HPL 参数扫描结果],
  )
]

在 4 rank 扫描中，最优组合为 `N=2000`、`NB=64`、`P x Q=1 x 4`，性能约为 `18.91 GFLOPS`。增大 `N` 后，计算量相对 MPI 启动和通信开销更占主导，因此结果比默认小规模测试更有意义。`NB=64` 在本环境中表现较好，继续增大块大小没有带来收益，说明较大的块降低了调度灵活性，而参考 BLAS 的局部计算效率不足以抵消这种影响。

进一步增大到 `N=4000` 后，使用 4 rank 和硬件线程绑定得到 `18.12 GFLOPS`。随后将 MPI rank 数增加到 8，并比较不同进程网格，结果如下：

#align(center)[
  #figure(
    table(
      columns: (auto, auto, auto, auto, auto, auto, auto),
      align: center + horizon,
      stroke: none,

      table.hline(stroke: 1pt),
      table.header([`N`], [`NB`], [`P x Q`], [MPI ranks], [绑定], [时间], [性能]),
      table.hline(stroke: 0.5pt),

      [`4000`], [`64`], [`1 x 4`], [`4`], [`hwthread`], [`2.36 s`], [`18.12 GFLOPS`],
      [`4000`], [`64`], [`1 x 8`], [`8`], [`hwthread`], [`2.19 s`], [`19.48 GFLOPS`],
      [`4000`], [`64`], [`2 x 4`], [`8`], [`hwthread`], [`2.00 s`], [`21.35 GFLOPS`],
      [`4000`], [`64`], [`4 x 2`], [`8`], [`hwthread`], [`1.94 s`], [`22.03 GFLOPS`],

      table.hline(stroke: 1pt),
    ),
    caption: [扩大规模与 8 rank 进程网格结果],
  )
]

最优普通 HPL 参数为 `N=4000`、`NB=64`、`P x Q=4 x 2`、8 个 MPI rank，并使用 `--bind-to hwthread --map-by hwthread` 绑定。该组合达到 `22.03 GFLOPS`，相比 4 rank 的 `N=4000, P x Q=1 x 4` 结果 `18.12 GFLOPS` 提升约 `21.6%`。提升不是线性的，原因是 HPL 中面板分解和广播通信不能随 rank 数完全等比例缩短，同时原始构建使用的参考 BLAS 限制了单 rank 的本地矩阵运算速度。

调优记录还比较了 CPU 绑定策略。在固定 `N=2000`、`NB=64`、`P x Q=1 x 4`、4 rank 时，默认运行约 `15.22 GFLOPS`，禁用绑定下降到 `13.34 GFLOPS`，按 core 绑定约 `15.17 GFLOPS`，按 hwthread 绑定约 `16.06 GFLOPS`。这说明在短作业中绑定能够减少 rank 迁移带来的缓存扰动，但差距不大，长时间测试仍应重复运行取平均值。

综合调优结果，本实验中最优的普通参考 BLAS 组合为 `N=4000, NB=64, P x Q=4 x 2, MPI ranks=8, hwthread binding`，性能为 `22.03 GFLOPS`。

= Bonus
== 更换数学库
#v(0.5em)

本实验完成了数学库更换的 Bonus。以下编译路径为 WSL 主机路径（`/home/cx/workspace/lab1/`），HPL 源码编译在 WSL 主机上完成，编译产物通过共享目录分发给容器节点。原始 HPL 构建使用前面从源码编译的参考 CBLAS/BLAS 静态库：

#codeblock(```make
LAinc = -I/home/cx/workspace/lab1/CBLAS/include
LAlib = /home/cx/workspace/lab1/CBLAS/lib/cblas_LINUX.a /home/cx/workspace/lab1/BLAS-3.12.0/blas_LINUX.a -lgfortran -lm
HPL_OPTS = -DHPL_CALL_CBLAS
```)

更换后的数学库为 BLIS 0.9.0。BLIS 是面向高性能 BLAS 的实现，提供更好的缓存分块、数据打包和 SIMD 微内核。本实验新建 `Make.BLIS`，将 HPL 链接到系统 BLIS 动态库：

#codeblock(```make
ARCH = BLIS
TOPdir = /home/cx/workspace/lab1/hpl-2.3
MPdir = /usr
MPlib = -L$(MPdir)/lib/x86_64-linux-gnu -lmpi
LAdir = /usr
LAlib = -L/usr/lib/x86_64-linux-gnu -lblis -lm -lpthread
HPL_OPTS = -DHPL_CALL_CBLAS
CC = mpicc
LINKER = mpicc
```)

BLIS 软件包和构建配置：

#figure(
  image("assets/lab1/image-24.png")
)

编译 BLIS 版本 HPL，并检查动态链接结果：

#figure(
  image("assets/lab1/image-25.png")
)


已有调优记录中，两种版本均使用 `N=4000`、`NB=64`、`P x Q=4 x 2`、8 个 MPI rank，并固定 `OMP_NUM_THREADS=1`、`BLIS_NUM_THREADS=1`，避免 BLAS 线程和 MPI 进程过度并行。以下数据为前期 WSL 主机调优时记录，`Make.BlIS` 和 `bin/BLIS/xhpl` 已验证存在（`ldd` 确认链接 `libblis.so.4`），但性能数值未在本次检查中重新复测。三次重复运行结果如下：

#align(center)[
  #figure(
    table(
      columns: (auto, auto, auto, auto, auto, auto),
      align: center + horizon,
      stroke: none,

      table.hline(stroke: 1pt),
      table.header([版本], [数学库], [运行次数], [平均时间], [平均性能], [相对速度]),
      table.hline(stroke: 0.5pt),

      [Original], [参考 CBLAS/BLAS], [`3`], [`2.09 s`], [`20.50 GFLOPS`], [`1.00x`],
      [BLIS], [BLIS 0.9.0], [`3`], [`0.73 s`], [`59.05 GFLOPS`], [`2.88x`],

      table.hline(stroke: 1pt),
    ),
    caption: [更换 BLIS 前后的 HPL 性能对比],
  )
]

BLIS 版本的平均性能达到 `59.05 GFLOPS`，约为原始参考 BLAS 版本 `20.50 GFLOPS` 的 `2.88` 倍。残差校验均通过，说明性能提升来自数学库实现差异，而不是错误计算。该结果表明 HPL 的主要耗时集中在 BLAS Level 3 运算中，更换优化 BLAS 是提升 HPL 性能最有效的方式之一。

== 使用共享文件系统运行 HPL
#v(0.5em)

除前文使用 NFS 外，额外在现有 Docker 集群中部署了一个最小可用 CephFS，用于验证分布式共享文件系统的基本读写能力。

CephFS 部署在原有 `10.42.0.0/24` 容器网络中。`node01` 作为 Ceph 服务端，运行 `mon`、`mgr`、`mds` 和一个基于 2 GiB loopback 文件的 `osd.0`；`node02`、`node03` 作为 CephFS 客户端，通过 `ceph-fuse` 挂载同一个文件系统到 `/mnt/cephfs`。该部署是单节点单 OSD 的实验验证环境，不具备高可用能力，但可以说明 CephFS 能够提供跨节点一致的共享目录。

在 `node02`、`node03` 上挂载 CephFS 并查看容量：

#figure(
  image("assets/lab1/image-28.png")
)

两个客户端均成功挂载。跨节点读写验证：

#figure(
  image("assets/lab1/image-29.png")
)

`node02` 写入的 `from-node02.txt` 可以在 `node03` 读取，`node03` 写入的 `from-node03.txt` 也可以在 `node02` 读取，说明两个客户端访问的是同一个 CephFS 文件系统。

Ceph 集群状态：

#figure(
  image("assets/lab1/image-27.png")
)

== K3s 基础部署
#v(0.5em)

K3s 部署在同一组 Docker 节点上，其中 `node01` 运行 k3s server，`node02`、`node03`、`node04` 作为 k3s agent 加入集群。由于 WSL/Docker-in-Docker 场景中容器内 overlayfs 不可用，启动 k3s 时使用 `--snapshotter native`；同时容器需要以 `--privileged` 和 `--cgroupns=host` 方式运行，才能满足 kubelet 对 cgroup 和网络能力的要求。

检查部署脚本和关键启动参数：

#figure(
  image("assets/lab1/image-30.png")
)

`k3s-setup.sh` 中 server 的核心启动参数如下：

#codeblock(```bash
k3s server \
  --node-ip 10.42.0.11 \
  --node-external-ip 10.42.0.11 \
  --flannel-iface eth0 \
  --cluster-cidr 10.244.0.0/16 \
  --pause-image local-pause:3.6 \
  --snapshotter native \
  --write-kubeconfig-mode 0644 \
  --disable traefik
```)

agent 节点使用 `node01` 生成的 token 加入集群，关键参数如下：

#codeblock(```bash
k3s agent \
  --server https://node01:6443 \
  --token <node-token> \
  --node-ip <node-ip> \
  --node-external-ip <node-ip> \
  --flannel-iface eth0 \
  --pause-image local-pause:3.6 \
  --snapshotter native
```)

部署和验证 K3s：

#figure(
  image("assets/lab1/image-31.png", width: 80%)
)
#figure(
  image("assets/lab1/image-32.png")
)

查看节点、Pod 和 Service：

#figure(
  image("assets/lab1/image-33.png")
)

`kubectl get nodes -o wide` 显示 `node01`、`node02`、`node03` 三个节点均为 `Ready`。

通过 NodePort 访问测试应用：

#figure(
  image("assets/lab1/image-34.png")
)

= AI Agent 使用说明
#v(0.5em)
由于时间有限，我使用 gpt + codex（plus 套餐）辅助完成实验报告。gpt 用来解答知识性问题和安装文件，codex 用来做 bonus，debug，生成验证脚本以及撰写实验报告。

codex 使用 docker 完成大部分实验，依本人的水平出错也无法核验。但 AI Agent 的特性是，只要目标明确，token 充足，就可以完成任务。在完成 lab1 的过程中，笔者的 codex 在 K3s 部署部分遇到挑战，花了相当久的时间才部署完成（也和网络不稳定有关）。最近实验压力较大。暑假，为了更好地理解 HPC 以及了解基本的 coding 命令，会适当减少 AI Agent 的暴力使用。

= 附录

== Slurm 作业脚本
#v(0.5em)
本实验中主要使用两个 Slurm 作业脚本。第一个脚本用于验证 Slurm 能够完成基本调度和执行，文件路径为 `/cluster/shared/slurm-test.sbatch`：

#codeblock(```bash
#!/bin/bash
#SBATCH --job-name=slurm-test
#SBATCH --partition=debug
#SBATCH --nodes=1
#SBATCH --ntasks=1
#SBATCH --output=/cluster/shared/slurm-test-%j.out

hostname
pwd
srun hostname
```)

第二个脚本用于通过 Slurm 启动 HPL，文件路径为 `/cluster/shared/hpl/run-hpl.sbatch`：

#codeblock(```bash
#!/bin/bash
#SBATCH --job-name=hpl
#SBATCH --partition=debug
#SBATCH --nodes=3
#SBATCH --ntasks-per-node=1
#SBATCH --output=/cluster/shared/hpl/hpl-%j.out
#SBATCH --error=/cluster/shared/hpl/hpl-%j.err

set -euo pipefail
cd /cluster/shared/hpl
srun --mpi=pmix -N3 --ntasks=3 ./xhpl
```)

提交作业时使用 `sbatch --parsable` 获取作业号，再按作业号读取输出文件，避免固定写死历史 job id：

#codeblock(```bash
jobid=$(sbatch --parsable /cluster/shared/hpl/run-hpl.sbatch)
echo "Submitted batch job ${jobid}"
squeue -j "${jobid}"
cat /cluster/shared/hpl/hpl-${jobid}.out
```)

== HPL 输出摘要

#codeblock(```text
N = 120
NB = 32
P x Q = 1 x 3
MPI ranks = 3
Output file = /cluster/shared/hpl/hpl-4.out

T/V                N    NB     P     Q               Time                 Gflops
WR00R2R2         120    32     1     3               0.01             1.2523e-01
||Ax-b||_oo/(eps*(||A||_oo*||x||_oo+||b||_oo)*N)=   1.34012681e-02 ...... PASSED
End of Tests.
```)

WSL 本地参数调优中的最优普通参考 BLAS 结果为：

#codeblock(```text
N = 4000
NB = 64
P x Q = 4 x 2
MPI ranks = 8
MPI binding = --bind-to hwthread --map-by hwthread
Observed performance = 22.03 GFLOPS
Residual check = PASSED
```)

更换 BLIS 数学库后的对比结果为：

#codeblock(```text
Original reference BLAS:
Average time = 2.09 s
Average performance = 20.50 GFLOPS
Relative speed = 1.00x

BLIS 0.9.0:
Average time = 0.73 s
Average performance = 59.05 GFLOPS
Relative speed = 2.88x
```)

== 修改文件说明
#v(0.5em)
本实验主要修改和新增了以下文件：

#align(center)[
  #figure(
    table(
      columns: (auto, auto, auto),
      align: center + horizon,
      stroke: none,

      table.hline(stroke: 1pt),
      table.header([文件], [修改内容], [原因]),
      table.hline(stroke: 0.5pt),

      [`/etc/hosts`], [加入 `node01` 至 `node04` 的静态解析], [保证 MPI、NFS、Slurm、CephFS 和 K3s 可通过节点名通信],
      [`/etc/exports`], [导出 `/cluster/shared` 给 `10.42.0.0/24`], [提供 NFS 共享目录],
      [`/etc/slurm/slurm.conf`], [配置 `node01` 为控制节点，`node02` 至 `node04` 为计算节点], [启用 Slurm `debug` 分区],
      [`BLAS-3.12.0/make.inc`], [设置 `FORTRAN=gfortran`、`FFLAGS=-O3 -fallow-argument-mismatch`], [从源码编译参考 BLAS 静态库],
      [`CBLAS/Makefile.in`], [设置 `BLLIB` 指向 `blas_LINUX.a`、`FC=gfortran`], [从源码编译 CBLAS 静态库],
      [`hpl-2.3/Make.Linux`], [设置 `LAlib` 链接 CBLAS/BLAS 静态库、`CC=mpicc`], [从源码编译 HPL（参考 BLAS 版本）],
      [`hpl-2.3/Make.BlIS`], [设置 `LAlib=-lblis` 动态链接 BLIS], [Bonus: 更换数学库为 BLIS],
      [`/cluster/shared/slurm-test.sbatch`], [新增 Slurm 调度验证脚本], [验证作业提交和 `srun` 执行],
      [`/cluster/shared/hpl/run-hpl.sbatch`], [新增 HPL 批处理脚本], [通过 Slurm 启动 HPL],
      [`/cluster/shared/hpl/HPL.dat`], [设置 `N=120, NB=32, P=1, Q=3`], [匹配 3 个 MPI rank 的集群验证运行],

      table.hline(stroke: 1pt),
    ),
    caption: [基础配置与 Slurm/HPL 文件],
  )
]

#align(center)[
  #figure(
    table(
      columns: (auto, auto, auto),
      align: center + horizon,
      stroke: none,

      table.hline(stroke: 1pt),
      table.header([文件], [修改内容], [原因]),
      table.hline(stroke: 0.5pt),

      [`/home/cx/workspace/lab1/hpl-2.3/Make.BLIS`], [新增 BLIS 链接配置], [对比参考 BLAS 与 BLIS 的 HPL 性能],
      [`wsl-cluster-lab/cephfs/bootstrap-node01.sh`], [创建 Ceph mon/mgr/mds/osd 和 CephFS], [验证 CephFS 共享文件系统],
      [`wsl-cluster-lab/cephfs/mount-client.sh`], [使用 `ceph-fuse` 挂载 `/mnt/cephfs`], [验证客户端挂载和跨节点读写],
      [`wsl-cluster-lab/k3s-setup.sh`], [启动 k3s server/agent 并使用 `--snapshotter native`], [适配 WSL/Docker-in-Docker 环境],
      [`wsl-cluster-lab/k3s-verify.sh`], [创建 `k3s-demo` Deployment 和 NodePort Service], [验证 Pod 调度和服务访问],

      table.hline(stroke: 1pt),
    ),
    caption: [Bonus 部分脚本与构建文件],
  )
]

其中，HPL 更换数学库时没有修改 HPL 源码，只新增了 `Make.BLIS` 并重新编译生成 `bin/BLIS/xhpl`。CephFS 和 K3s 部分均为实验环境脚本，用于在 Docker 容器节点中复现实验配置。
