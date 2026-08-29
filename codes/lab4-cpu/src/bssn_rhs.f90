

#include "macrodef.fh"

  function compute_rhs_bssn(ex, T,X, Y, Z,                                     &
               chi    ,   trK    ,                                             &
               dxx    ,   gxy    ,   gxz    ,   dyy    ,   gyz    ,   dzz,     &
               Axx    ,   Axy    ,   Axz    ,   Ayy    ,   Ayz    ,   Azz,     &
               Gamx   ,  Gamy    ,  Gamz    ,                                  &
               Lap    ,  betax   ,  betay   ,  betaz   ,                       &
               dtSfx  ,  dtSfy   ,  dtSfz   ,                                  &
               chi_rhs,   trK_rhs,                                             &
               gxx_rhs,   gxy_rhs,   gxz_rhs,   gyy_rhs,   gyz_rhs,   gzz_rhs, &
               Axx_rhs,   Axy_rhs,   Axz_rhs,   Ayy_rhs,   Ayz_rhs,   Azz_rhs, &
               Gamx_rhs,  Gamy_rhs,  Gamz_rhs,                                 &
               Lap_rhs,  betax_rhs,  betay_rhs,  betaz_rhs,                    &
               dtSfx_rhs,  dtSfy_rhs,  dtSfz_rhs,                              &
               rho,Sx,Sy,Sz,Sxx,Sxy,Sxz,Syy,Syz,Szz,                           &
               Gamxxx,Gamxxy,Gamxxz,Gamxyy,Gamxyz,Gamxzz,                      &
               Gamyxx,Gamyxy,Gamyxz,Gamyyy,Gamyyz,Gamyzz,                      &
               Gamzxx,Gamzxy,Gamzxz,Gamzyy,Gamzyz,Gamzzz,                      &
               Rxx,Rxy,Rxz,Ryy,Ryz,Rzz,                                        &
               ham_Res, movx_Res, movy_Res, movz_Res,                          &
                        Gmx_Res, Gmy_Res, Gmz_Res,                             &
               Symmetry,Lev,eps,co)  result(gont)
! calculate constraint violation when co=0               
  implicit none

!~~~~~~> Input parameters:

  integer,intent(in ):: ex(1:3), Symmetry,Lev,co
  integer :: i,j,k  ! AMSS_ENABLE_RHS_GIJ_LOOPS
  real*8, intent(in ):: T
  real*8, intent(in ):: X(1:ex(1)),Y(1:ex(2)),Z(1:ex(3))
  real*8, dimension(ex(1),ex(2),ex(3)),intent(inout) :: chi,dxx,dyy,dzz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ) :: trK
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ) :: gxy,gxz,gyz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ) :: Axx,Axy,Axz,Ayy,Ayz,Azz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ) :: Gamx,Gamy,Gamz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(inout) :: Lap, betax, betay, betaz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ) :: dtSfx,  dtSfy,  dtSfz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: chi_rhs,trK_rhs
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: gxx_rhs,gxy_rhs,gxz_rhs
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: gyy_rhs,gyz_rhs,gzz_rhs
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: Axx_rhs,Axy_rhs,Axz_rhs
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: Ayy_rhs,Ayz_rhs,Azz_rhs
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: Gamx_rhs,Gamy_rhs,Gamz_rhs
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: Lap_rhs, betax_rhs, betay_rhs, betaz_rhs
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: dtSfx_rhs,dtSfy_rhs,dtSfz_rhs
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ) :: rho,Sx,Sy,Sz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(in ) :: Sxx,Sxy,Sxz,Syy,Syz,Szz
! when out, physical second kind of connection  
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: Gamxxx, Gamxxy, Gamxxz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: Gamxyy, Gamxyz, Gamxzz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: Gamyxx, Gamyxy, Gamyxz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: Gamyyy, Gamyyz, Gamyzz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: Gamzxx, Gamzxy, Gamzxz
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: Gamzyy, Gamzyz, Gamzzz
! when out, physical Ricci tensor  
  real*8, dimension(ex(1),ex(2),ex(3)),intent(out) :: Rxx,Rxy,Rxz,Ryy,Ryz,Rzz
  real*8,intent(in) :: eps
  real*8, dimension(ex(1),ex(2),ex(3)),intent(inout) :: ham_Res, movx_Res, movy_Res, movz_Res
  real*8, dimension(ex(1),ex(2),ex(3)),intent(inout) :: Gmx_Res, Gmy_Res, Gmz_Res
!  gont = 0: success; gont = 1: something wrong
  integer::gont

!~~~~~~> Other variables:

  real*8, dimension(ex(1),ex(2),ex(3)) :: gxx,gyy,gzz
  real*8, dimension(ex(1),ex(2),ex(3)) :: chix,chiy,chiz
  real*8, dimension(ex(1),ex(2),ex(3)) :: gxxx,gxyx,gxzx,gyyx,gyzx,gzzx
  real*8, dimension(ex(1),ex(2),ex(3)) :: gxxy,gxyy,gxzy,gyyy,gyzy,gzzy
  real*8, dimension(ex(1),ex(2),ex(3)) :: gxxz,gxyz,gxzz,gyyz,gyzz,gzzz
  real*8, dimension(ex(1),ex(2),ex(3)) :: Lapx,Lapy,Lapz
  real*8, dimension(ex(1),ex(2),ex(3)) :: betaxx,betaxy,betaxz
  real*8, dimension(ex(1),ex(2),ex(3)) :: betayx,betayy,betayz
  real*8, dimension(ex(1),ex(2),ex(3)) :: betazx,betazy,betazz
  real*8, dimension(ex(1),ex(2),ex(3)) :: Gamxx,Gamxy,Gamxz
  real*8, dimension(ex(1),ex(2),ex(3)) :: Gamyx,Gamyy,Gamyz
  real*8, dimension(ex(1),ex(2),ex(3)) :: Gamzx,Gamzy,Gamzz
  real*8, dimension(ex(1),ex(2),ex(3)) :: Kx,Ky,Kz
  real*8, dimension(ex(1),ex(2),ex(3)) :: fxx,fxy,fxz,fyy,fyz,fzz
  real*8, dimension(ex(1),ex(2),ex(3)) :: Gamxa,Gamya,Gamza
  real*8 :: alpn1, chin1, div_beta, det, f, S
  real*8 :: s_gxx, s_gyy, s_gzz
  real*8 :: s_fxx, s_fxy, s_fxz, s_fyy, s_fyz, s_fzz
  real*8 :: s_gxxx, s_gxxy, s_gxxz
  real*8 :: hm, mx, my, mz
  real*8, dimension(ex(1),ex(2),ex(3)) :: gupxx,gupxy,gupxz
  real*8, dimension(ex(1),ex(2),ex(3)) :: gupyy,gupyz,gupzz

  real*8,dimension(3) ::SSS,AAS,ASA,SAA,ASS,SAS,SSA
  real*8            :: dX, dY, dZ, PI
  real*8, parameter :: ZEO = 0.d0,ONE = 1.D0, TWO = 2.D0, FOUR = 4.D0
  real*8, parameter :: EIGHT = 8.D0, HALF = 0.5D0, THR = 3.d0
  real*8, parameter :: SYM = 1.D0, ANTI= - 1.D0
  double precision,parameter::FF = 0.75d0,eta=2.d0
  real*8, parameter :: F1o3 = 1.D0/3.D0, F2o3 = 2.D0/3.D0,F3o2=1.5d0, F1o6 = 1.D0/6.D0
  real*8, parameter :: F16=1.6d1,F8=8.d0



  real*8, dimension(ex(1),ex(2)) :: p_betaxx, p_betaxy, p_betaxz, p_betayx, p_betayy, p_betayz
  real*8, dimension(ex(1),ex(2)) :: p_betazx, p_betazy, p_betazz, p_chix, p_chiy, p_chiz
  real*8, dimension(ex(1),ex(2)) :: p_gxxx, p_gxyx, p_gxzx, p_gyyx, p_gyzx, p_gzzx
  real*8, dimension(ex(1),ex(2)) :: p_gxxy, p_gxyy, p_gxzy, p_gyyy, p_gyzy, p_gzzy
  real*8, dimension(ex(1),ex(2)) :: p_gxxz, p_gxyz, p_gxzz, p_gyyz, p_gyzz, p_gzzz
  real*8, dimension(ex(1),ex(2)) :: p_Lapx, p_Lapy, p_Lapz, p_Kx, p_Ky, p_Kz
  real*8, dimension(ex(1),ex(2)) :: p_Gamxx, p_Gamxy, p_Gamxz, p_Gamyx, p_Gamyy, p_Gamyz
  real*8, dimension(ex(1),ex(2)) :: p_Gamzx, p_Gamzy, p_Gamzz
  real*8, dimension(ex(1),ex(2)) :: p_fxx, p_fxy, p_fxz, p_fyy, p_fyz, p_fzz
  real*8, dimension(ex(1),ex(2)) :: p_gupxx, p_gupxy, p_gupxz, p_gupyy, p_gupyz, p_gupzz
  real*8, dimension(ex(1),ex(2)) :: p_Gamxa, p_Gamya, p_Gamza
  real*8 :: d12dx, d12dy, d12dz, d2dx, d2dy, d2dz
  real*8 :: Sdxdx, Sdydy, Sdzdz, Fdxdx, Fdydy, Fdzdz
  real*8 :: Sdxdy, Sdxdz, Sdydz, Fdxdy, Fdxdz, Fdydz
  integer :: imin, jmin, kmin, imax, jmax, kmax
  logical :: k4, k2
  real*8, parameter :: F1o12 = ONE/1.2d1, F1o4 = 2.5d-1, F1o144 = ONE/1.44d2
  real*8, parameter :: F12 = 1.2d1, F30 = 3.d1, EIT = 8.d0

!!! sanity check
#ifdef AMSS_RHS_NAN_CHECK
  dX = sum(chi)+sum(trK)+sum(dxx)+sum(gxy)+sum(gxz)+sum(dyy)+sum(gyz)+sum(dzz) &
      +sum(Axx)+sum(Axy)+sum(Axz)+sum(Ayy)+sum(Ayz)+sum(Azz)                   &
      +sum(Gamx)+sum(Gamy)+sum(Gamz)                                           &
      +sum(Lap)+sum(betax)+sum(betay)+sum(betaz)
  if(dX.ne.dX) then
     if(sum(chi).ne.sum(chi))write(*,*)"bssn.f90: find NaN in chi"
     if(sum(trK).ne.sum(trK))write(*,*)"bssn.f90: find NaN in trk"
     if(sum(dxx).ne.sum(dxx))write(*,*)"bssn.f90: find NaN in dxx"
     if(sum(gxy).ne.sum(gxy))write(*,*)"bssn.f90: find NaN in gxy"
     if(sum(gxz).ne.sum(gxz))write(*,*)"bssn.f90: find NaN in gxz"
     if(sum(dyy).ne.sum(dyy))write(*,*)"bssn.f90: find NaN in dyy"
     if(sum(gyz).ne.sum(gyz))write(*,*)"bssn.f90: find NaN in gyz"
     if(sum(dzz).ne.sum(dzz))write(*,*)"bssn.f90: find NaN in dzz"
     if(sum(Axx).ne.sum(Axx))write(*,*)"bssn.f90: find NaN in Axx"
     if(sum(Axy).ne.sum(Axy))write(*,*)"bssn.f90: find NaN in Axy"
     if(sum(Axz).ne.sum(Axz))write(*,*)"bssn.f90: find NaN in Axz"
     if(sum(Ayy).ne.sum(Ayy))write(*,*)"bssn.f90: find NaN in Ayy"
     if(sum(Ayz).ne.sum(Ayz))write(*,*)"bssn.f90: find NaN in Ayz"
     if(sum(Azz).ne.sum(Azz))write(*,*)"bssn.f90: find NaN in Azz"
     if(sum(Gamx).ne.sum(Gamx))write(*,*)"bssn.f90: find NaN in Gamx"
     if(sum(Gamy).ne.sum(Gamy))write(*,*)"bssn.f90: find NaN in Gamy"
     if(sum(Gamz).ne.sum(Gamz))write(*,*)"bssn.f90: find NaN in Gamz"
     if(sum(Lap).ne.sum(Lap))write(*,*)"bssn.f90: find NaN in Lap"
     if(sum(betax).ne.sum(betax))write(*,*)"bssn.f90: find NaN in betax"
     if(sum(betay).ne.sum(betay))write(*,*)"bssn.f90: find NaN in betay"
     if(sum(betaz).ne.sum(betaz))write(*,*)"bssn.f90: find NaN in betaz"
     gont = 1
     return
  endif
#endif

  PI = dacos(-ONE)

  dX = X(2) - X(1)
  dY = Y(2) - Y(1)
  dZ = Z(2) - Z(1)

  ! alpn1/chin1 become scalars inside fused loops (Stage 1a)
  gxx = dxx + ONE
  gyy = dyy + ONE
  gzz = dzz + ONE

  imax = ex(1)
  jmax = ex(2)
  kmax = ex(3)
  imin = 1
  jmin = 1
  kmin = 1
  if(Symmetry > 0 .and. dabs(Z(1)) < dZ) kmin = -1
  if(Symmetry > 1 .and. dabs(X(1)) < dX) imin = -1
  if(Symmetry > 1 .and. dabs(Y(1)) < dY) jmin = -1

  d12dx = ONE/F12/dX
  d12dy = ONE/F12/dY
  d12dz = ONE/F12/dZ
  d2dx = ONE/TWO/dX
  d2dy = ONE/TWO/dY
  d2dz = ONE/TWO/dZ

  Sdxdx = ONE/(dX*dX)
  Sdydy = ONE/(dY*dY)
  Sdzdz = ONE/(dZ*dZ)
  Fdxdx = F1o12/(dX*dX)
  Fdydy = F1o12/(dY*dY)
  Fdzdz = F1o12/(dZ*dZ)
  Sdxdy = F1o4/(dX*dY)
  Sdxdz = F1o4/(dX*dZ)
  Sdydz = F1o4/(dY*dZ)
  Fdxdy = F1o144/(dX*dY)
  Fdxdz = F1o144/(dX*dZ)
  Fdydz = F1o144/(dY*dZ)

! ==================== Stage 1b: k-rolling-window fusion ====================
! All 21 fderivs + 11 fdderivs calls become per-k-plane computations; the 60
! 3D intermediate arrays (betaxx.., chix.., gxxx.., Lapx.., Kx.., Gamxx..,
! Gamxa.., fxx.., gupxx..) become 2D planes p_*(ex(1),ex(2)) per k. Boundary
! semantics (joint 4th/2nd-order conditions, symmetry reflection, k=ex(3)
! zero plane) replicate fderivs/fdderivs exactly; per-point arithmetic order
! is preserved so results remain bit-exact vs the whole-array baseline.
  do k = 1, ex(3)
    k4 = (k+2 .le. kmax .and. k-2 .ge. kmin)
    k2 = (k+1 .le. kmax .and. k-1 .ge. kmin)
  call fderivs_plane(betax, p_betaxx, p_betaxy, p_betaxz, ANTI, SYM, SYM)
  call fderivs_plane(betay, p_betayx, p_betayy, p_betayz, SYM, ANTI, SYM)
  call fderivs_plane(betaz, p_betazx, p_betazy, p_betazz, SYM, SYM, ANTI)
  call fderivs_plane(chi, p_chix, p_chiy, p_chiz, SYM, SYM, SYM)
  do j = 1, ex(2)
  do i = 1, ex(1)
  alpn1   = Lap(i,j,k) + ONE
  chin1   = chi(i,j,k) + ONE
  div_beta = p_betaxx(i,j) + p_betayy(i,j) + p_betazz(i,j)
  chi_rhs(i,j,k) = F2o3 * chin1 * ( alpn1 * trK(i,j,k) - div_beta ) !rhs for chi
  end do
  end do
  call fderivs_plane(dxx, p_gxxx, p_gxxy, p_gxxz, SYM, SYM, SYM)
  call fderivs_plane(gxy, p_gxyx, p_gxyy, p_gxyz, ANTI, ANTI, SYM)
  call fderivs_plane(gxz, p_gxzx, p_gxzy, p_gxzz, ANTI, SYM, ANTI)
  call fderivs_plane(dyy, p_gyyx, p_gyyy, p_gyyz, SYM, SYM, SYM)
  call fderivs_plane(gyz, p_gyzx, p_gyzy, p_gyzz, SYM, ANTI, ANTI)
  call fderivs_plane(dzz, p_gzzx, p_gzzy, p_gzzz, SYM, SYM, SYM)
  do j = 1, ex(2)
  do i = 1, ex(1)
  alpn1   = Lap(i,j,k) + ONE
  s_gxx   = dxx(i,j,k) + ONE
  s_gyy   = dyy(i,j,k) + ONE
  s_gzz   = dzz(i,j,k) + ONE
  div_beta = p_betaxx(i,j) + p_betayy(i,j) + p_betazz(i,j)
  gxx_rhs(i,j,k) = - TWO * alpn1 * Axx(i,j,k) - F2o3 * s_gxx * div_beta + &
  TWO *(  s_gxx * p_betaxx(i,j) +   gxy(i,j,k) * p_betayx(i,j) +   gxz(i,j,k) * p_betazx(i,j))
  gyy_rhs(i,j,k) = - TWO * alpn1 * Ayy(i,j,k) - F2o3 * s_gyy * div_beta + &
  TWO *(  gxy(i,j,k) * p_betaxy(i,j) +   s_gyy * p_betayy(i,j) +   gyz(i,j,k) * p_betazy(i,j))
  gzz_rhs(i,j,k) = - TWO * alpn1 * Azz(i,j,k) - F2o3 * s_gzz * div_beta + &
  TWO *(  gxz(i,j,k) * p_betaxz(i,j) +   gyz(i,j,k) * p_betayz(i,j) +   s_gzz * p_betazz(i,j))
  gxy_rhs(i,j,k) = - TWO * alpn1 * Axy(i,j,k) +  F1o3 * gxy(i,j,k) * div_beta + &
  s_gxx * p_betaxy(i,j)                  +   gxz(i,j,k) * p_betazy(i,j) + &
  s_gyy * p_betayx(i,j) +   gyz(i,j,k) * p_betazx(i,j)   &
  -   gxy(i,j,k) * p_betazz(i,j)
  gyz_rhs(i,j,k) = - TWO * alpn1 * Ayz(i,j,k) +  F1o3 * gyz(i,j,k) * div_beta + &
  gxy(i,j,k) * p_betaxz(i,j) +   s_gyy * p_betayz(i,j)                  + &
  gxz(i,j,k) * p_betaxy(i,j)                  +   s_gzz * p_betazy(i,j)   &
  -   gyz(i,j,k) * p_betaxx(i,j)
  gxz_rhs(i,j,k) = - TWO * alpn1 * Axz(i,j,k) +  F1o3 * gxz(i,j,k) * div_beta + &
  s_gxx * p_betaxz(i,j) +   gxy(i,j,k) * p_betayz(i,j)                  + &
  gyz(i,j,k) * p_betayx(i,j) +   s_gzz * p_betazx(i,j)   &
  -   gxz(i,j,k) * p_betayy(i,j)     !rhs for gij
  ! invert tilted metric (det = determinant gupzz)
  det = s_gxx * s_gyy * s_gzz + gxy(i,j,k) * gyz(i,j,k) * gxz(i,j,k) + &
  gxz(i,j,k) * gxy(i,j,k) * gyz(i,j,k) - &
  gxz(i,j,k) * s_gyy * gxz(i,j,k) - gxy(i,j,k) * gxy(i,j,k) * s_gzz - &
  s_gxx * gyz(i,j,k) * gyz(i,j,k)
  p_gupxx(i,j) =   ( s_gyy * s_gzz - gyz(i,j,k) * gyz(i,j,k) ) / det
  p_gupxy(i,j) = - ( gxy(i,j,k) * s_gzz - gyz(i,j,k) * gxz(i,j,k) ) / det
  p_gupxz(i,j) =   ( gxy(i,j,k) * gyz(i,j,k) - s_gyy * gxz(i,j,k) ) / det
  p_gupyy(i,j) =   ( s_gxx * s_gzz - gxz(i,j,k) * gxz(i,j,k) ) / det
  p_gupyz(i,j) = - ( s_gxx * gyz(i,j,k) - gxy(i,j,k) * gxz(i,j,k) ) / det
  p_gupzz(i,j) =   ( s_gxx * s_gyy - gxy(i,j,k) * gxy(i,j,k) ) / det
  end do
  end do
    if (co == 0) then
    do j = 1, ex(2)
    do i = 1, ex(1)
    Gmx_Res(i,j,k) = Gamx(i,j,k) - (p_gupxx(i,j)*(p_gupxx(i,j)*p_gxxx(i,j)+p_gupxy(i,j)*p_gxyx(i,j)+p_gupxz(i,j)*p_gxzx(i,j))&
    +p_gupxy(i,j)*(p_gupxx(i,j)*p_gxyx(i,j)+p_gupxy(i,j)*p_gyyx(i,j)+p_gupxz(i,j)*p_gyzx(i,j))&
    +p_gupxz(i,j)*(p_gupxx(i,j)*p_gxzx(i,j)+p_gupxy(i,j)*p_gyzx(i,j)+p_gupxz(i,j)*p_gzzx(i,j))&
    +p_gupxx(i,j)*(p_gupxy(i,j)*p_gxxy(i,j)+p_gupyy(i,j)*p_gxyy(i,j)+p_gupyz(i,j)*p_gxzy(i,j))&
    +p_gupxy(i,j)*(p_gupxy(i,j)*p_gxyy(i,j)+p_gupyy(i,j)*p_gyyy(i,j)+p_gupyz(i,j)*p_gyzy(i,j))&
    +p_gupxz(i,j)*(p_gupxy(i,j)*p_gxzy(i,j)+p_gupyy(i,j)*p_gyzy(i,j)+p_gupyz(i,j)*p_gzzy(i,j))&
    +p_gupxx(i,j)*(p_gupxz(i,j)*p_gxxz(i,j)+p_gupyz(i,j)*p_gxyz(i,j)+p_gupzz(i,j)*p_gxzz(i,j))&
    +p_gupxy(i,j)*(p_gupxz(i,j)*p_gxyz(i,j)+p_gupyz(i,j)*p_gyyz(i,j)+p_gupzz(i,j)*p_gyzz(i,j))&
    +p_gupxz(i,j)*(p_gupxz(i,j)*p_gxzz(i,j)+p_gupyz(i,j)*p_gyzz(i,j)+p_gupzz(i,j)*p_gzzz(i,j)))
    Gmy_Res(i,j,k) = Gamy(i,j,k) - (p_gupxx(i,j)*(p_gupxy(i,j)*p_gxxx(i,j)+p_gupyy(i,j)*p_gxyx(i,j)+p_gupyz(i,j)*p_gxzx(i,j))&
    +p_gupxy(i,j)*(p_gupxy(i,j)*p_gxyx(i,j)+p_gupyy(i,j)*p_gyyx(i,j)+p_gupyz(i,j)*p_gyzx(i,j))&
    +p_gupxz(i,j)*(p_gupxy(i,j)*p_gxzx(i,j)+p_gupyy(i,j)*p_gyzx(i,j)+p_gupyz(i,j)*p_gzzx(i,j))&
    +p_gupxy(i,j)*(p_gupxy(i,j)*p_gxxy(i,j)+p_gupyy(i,j)*p_gxyy(i,j)+p_gupyz(i,j)*p_gxzy(i,j))&
    +p_gupyy(i,j)*(p_gupxy(i,j)*p_gxyy(i,j)+p_gupyy(i,j)*p_gyyy(i,j)+p_gupyz(i,j)*p_gyzy(i,j))&
    +p_gupyz(i,j)*(p_gupxy(i,j)*p_gxzy(i,j)+p_gupyy(i,j)*p_gyzy(i,j)+p_gupyz(i,j)*p_gzzy(i,j))&
    +p_gupxy(i,j)*(p_gupxz(i,j)*p_gxxz(i,j)+p_gupyz(i,j)*p_gxyz(i,j)+p_gupzz(i,j)*p_gxzz(i,j))&
    +p_gupyy(i,j)*(p_gupxz(i,j)*p_gxyz(i,j)+p_gupyz(i,j)*p_gyyz(i,j)+p_gupzz(i,j)*p_gyzz(i,j))&
    +p_gupyz(i,j)*(p_gupxz(i,j)*p_gxzz(i,j)+p_gupyz(i,j)*p_gyzz(i,j)+p_gupzz(i,j)*p_gzzz(i,j)))
    Gmz_Res(i,j,k) = Gamz(i,j,k) - (p_gupxx(i,j)*(p_gupxz(i,j)*p_gxxx(i,j)+p_gupyz(i,j)*p_gxyx(i,j)+p_gupzz(i,j)*p_gxzx(i,j))&
    +p_gupxy(i,j)*(p_gupxz(i,j)*p_gxyx(i,j)+p_gupyz(i,j)*p_gyyx(i,j)+p_gupzz(i,j)*p_gyzx(i,j))&
    +p_gupxz(i,j)*(p_gupxz(i,j)*p_gxzx(i,j)+p_gupyz(i,j)*p_gyzx(i,j)+p_gupzz(i,j)*p_gzzx(i,j))&
    +p_gupxy(i,j)*(p_gupxz(i,j)*p_gxxy(i,j)+p_gupyz(i,j)*p_gxyy(i,j)+p_gupzz(i,j)*p_gxzy(i,j))&
    +p_gupyy(i,j)*(p_gupxz(i,j)*p_gxyy(i,j)+p_gupyz(i,j)*p_gyyy(i,j)+p_gupzz(i,j)*p_gyzy(i,j))&
    +p_gupyz(i,j)*(p_gupxz(i,j)*p_gxzy(i,j)+p_gupyz(i,j)*p_gyzy(i,j)+p_gupzz(i,j)*p_gzzy(i,j))&
    +p_gupxz(i,j)*(p_gupxz(i,j)*p_gxxz(i,j)+p_gupyz(i,j)*p_gxyz(i,j)+p_gupzz(i,j)*p_gxzz(i,j))&
    +p_gupyz(i,j)*(p_gupxz(i,j)*p_gxyz(i,j)+p_gupyz(i,j)*p_gyyz(i,j)+p_gupzz(i,j)*p_gyzz(i,j))&
    +p_gupzz(i,j)*(p_gupxz(i,j)*p_gxzz(i,j)+p_gupyz(i,j)*p_gyzz(i,j)+p_gupzz(i,j)*p_gzzz(i,j)))
    end do
    end do
    end if
    do j = 1, ex(2)
      do i = 1, ex(1)
        Gamxxx(i,j,k) =HALF*( p_gupxx(i,j)*p_gxxx(i,j) + p_gupxy(i,j)*(TWO*p_gxyx(i,j) - p_gxxy(i,j) ) + p_gupxz(i,j)*(TWO*p_gxzx(i,j) - p_gxxz(i,j) ))
        Gamyxx(i,j,k) =HALF*( p_gupxy(i,j)*p_gxxx(i,j) + p_gupyy(i,j)*(TWO*p_gxyx(i,j) - p_gxxy(i,j) ) + p_gupyz(i,j)*(TWO*p_gxzx(i,j) - p_gxxz(i,j) ))
        Gamzxx(i,j,k) =HALF*( p_gupxz(i,j)*p_gxxx(i,j) + p_gupyz(i,j)*(TWO*p_gxyx(i,j) - p_gxxy(i,j) ) + p_gupzz(i,j)*(TWO*p_gxzx(i,j) - p_gxxz(i,j) ))

        Gamxyy(i,j,k) =HALF*( p_gupxx(i,j)*(TWO*p_gxyy(i,j) - p_gyyx(i,j) ) + p_gupxy(i,j)*p_gyyy(i,j) + p_gupxz(i,j)*(TWO*p_gyzy(i,j) - p_gyyz(i,j) ))
        Gamyyy(i,j,k) =HALF*( p_gupxy(i,j)*(TWO*p_gxyy(i,j) - p_gyyx(i,j) ) + p_gupyy(i,j)*p_gyyy(i,j) + p_gupyz(i,j)*(TWO*p_gyzy(i,j) - p_gyyz(i,j) ))
        Gamzyy(i,j,k) =HALF*( p_gupxz(i,j)*(TWO*p_gxyy(i,j) - p_gyyx(i,j) ) + p_gupyz(i,j)*p_gyyy(i,j) + p_gupzz(i,j)*(TWO*p_gyzy(i,j) - p_gyyz(i,j) ))

        Gamxzz(i,j,k) =HALF*( p_gupxx(i,j)*(TWO*p_gxzz(i,j) - p_gzzx(i,j) ) + p_gupxy(i,j)*(TWO*p_gyzz(i,j) - p_gzzy(i,j) ) + p_gupxz(i,j)*p_gzzz(i,j))
        Gamyzz(i,j,k) =HALF*( p_gupxy(i,j)*(TWO*p_gxzz(i,j) - p_gzzx(i,j) ) + p_gupyy(i,j)*(TWO*p_gyzz(i,j) - p_gzzy(i,j) ) + p_gupyz(i,j)*p_gzzz(i,j))
        Gamzzz(i,j,k) =HALF*( p_gupxz(i,j)*(TWO*p_gxzz(i,j) - p_gzzx(i,j) ) + p_gupyz(i,j)*(TWO*p_gyzz(i,j) - p_gzzy(i,j) ) + p_gupzz(i,j)*p_gzzz(i,j))

        Gamxxy(i,j,k) =HALF*( p_gupxx(i,j)*p_gxxy(i,j) + p_gupxy(i,j)*p_gyyx(i,j) + p_gupxz(i,j)*( p_gxzy(i,j) + p_gyzx(i,j) - p_gxyz(i,j) ) )
        Gamyxy(i,j,k) =HALF*( p_gupxy(i,j)*p_gxxy(i,j) + p_gupyy(i,j)*p_gyyx(i,j) + p_gupyz(i,j)*( p_gxzy(i,j) + p_gyzx(i,j) - p_gxyz(i,j) ) )
        Gamzxy(i,j,k) =HALF*( p_gupxz(i,j)*p_gxxy(i,j) + p_gupyz(i,j)*p_gyyx(i,j) + p_gupzz(i,j)*( p_gxzy(i,j) + p_gyzx(i,j) - p_gxyz(i,j) ) )

        Gamxxz(i,j,k) =HALF*( p_gupxx(i,j)*p_gxxz(i,j) + p_gupxy(i,j)*( p_gxyz(i,j) + p_gyzx(i,j) - p_gxzy(i,j) ) + p_gupxz(i,j)*p_gzzx(i,j) )
        Gamyxz(i,j,k) =HALF*( p_gupxy(i,j)*p_gxxz(i,j) + p_gupyy(i,j)*( p_gxyz(i,j) + p_gyzx(i,j) - p_gxzy(i,j) ) + p_gupyz(i,j)*p_gzzx(i,j) )
        Gamzxz(i,j,k) =HALF*( p_gupxz(i,j)*p_gxxz(i,j) + p_gupyz(i,j)*( p_gxyz(i,j) + p_gyzx(i,j) - p_gxzy(i,j) ) + p_gupzz(i,j)*p_gzzx(i,j) )

        Gamxyz(i,j,k) =HALF*( p_gupxx(i,j)*( p_gxyz(i,j) + p_gxzy(i,j) - p_gyzx(i,j) ) + p_gupxy(i,j)*p_gyyz(i,j) + p_gupxz(i,j)*p_gzzy(i,j) )
        Gamyyz(i,j,k) =HALF*( p_gupxy(i,j)*( p_gxyz(i,j) + p_gxzy(i,j) - p_gyzx(i,j) ) + p_gupyy(i,j)*p_gyyz(i,j) + p_gupyz(i,j)*p_gzzy(i,j) )
        Gamzyz(i,j,k) =HALF*( p_gupxz(i,j)*( p_gxyz(i,j) + p_gxzy(i,j) - p_gyzx(i,j) ) + p_gupyz(i,j)*p_gyyz(i,j) + p_gupzz(i,j)*p_gzzy(i,j) )
        ! Raise indices of \tilde A_{ij} and store in R_ij
      end do
    end do
    do j = 1, ex(2)
      do i = 1, ex(1)
        Rxx(i,j,k) =    p_gupxx(i,j) * p_gupxx(i,j) * Axx(i,j,k) + p_gupxy(i,j) * p_gupxy(i,j) * Ayy(i,j,k) + p_gupxz(i,j) * p_gupxz(i,j) * Azz(i,j,k) + &
        TWO*(p_gupxx(i,j) * p_gupxy(i,j) * Axy(i,j,k) + p_gupxx(i,j) * p_gupxz(i,j) * Axz(i,j,k) + p_gupxy(i,j) * p_gupxz(i,j) * Ayz(i,j,k))

        Ryy(i,j,k) =    p_gupxy(i,j) * p_gupxy(i,j) * Axx(i,j,k) + p_gupyy(i,j) * p_gupyy(i,j) * Ayy(i,j,k) + p_gupyz(i,j) * p_gupyz(i,j) * Azz(i,j,k) + &
        TWO*(p_gupxy(i,j) * p_gupyy(i,j) * Axy(i,j,k) + p_gupxy(i,j) * p_gupyz(i,j) * Axz(i,j,k) + p_gupyy(i,j) * p_gupyz(i,j) * Ayz(i,j,k))

        Rzz(i,j,k) =    p_gupxz(i,j) * p_gupxz(i,j) * Axx(i,j,k) + p_gupyz(i,j) * p_gupyz(i,j) * Ayy(i,j,k) + p_gupzz(i,j) * p_gupzz(i,j) * Azz(i,j,k) + &
        TWO*(p_gupxz(i,j) * p_gupyz(i,j) * Axy(i,j,k) + p_gupxz(i,j) * p_gupzz(i,j) * Axz(i,j,k) + p_gupyz(i,j) * p_gupzz(i,j) * Ayz(i,j,k))

        Rxy(i,j,k) =    p_gupxx(i,j) * p_gupxy(i,j) * Axx(i,j,k) + p_gupxy(i,j) * p_gupyy(i,j) * Ayy(i,j,k) + p_gupxz(i,j) * p_gupyz(i,j) * Azz(i,j,k) + &
        (p_gupxx(i,j) * p_gupyy(i,j)       + p_gupxy(i,j) * p_gupxy(i,j))* Axy(i,j,k)                       + &
        (p_gupxx(i,j) * p_gupyz(i,j)       + p_gupxz(i,j) * p_gupxy(i,j))* Axz(i,j,k)                       + &
        (p_gupxy(i,j) * p_gupyz(i,j)       + p_gupxz(i,j) * p_gupyy(i,j))* Ayz(i,j,k)

        Rxz(i,j,k) =    p_gupxx(i,j) * p_gupxz(i,j) * Axx(i,j,k) + p_gupxy(i,j) * p_gupyz(i,j) * Ayy(i,j,k) + p_gupxz(i,j) * p_gupzz(i,j) * Azz(i,j,k) + &
        (p_gupxx(i,j) * p_gupyz(i,j)       + p_gupxy(i,j) * p_gupxz(i,j))* Axy(i,j,k)                       + &
        (p_gupxx(i,j) * p_gupzz(i,j)       + p_gupxz(i,j) * p_gupxz(i,j))* Axz(i,j,k)                       + &
        (p_gupxy(i,j) * p_gupzz(i,j)       + p_gupxz(i,j) * p_gupyz(i,j))* Ayz(i,j,k)

        Ryz(i,j,k) =    p_gupxy(i,j) * p_gupxz(i,j) * Axx(i,j,k) + p_gupyy(i,j) * p_gupyz(i,j) * Ayy(i,j,k) + p_gupyz(i,j) * p_gupzz(i,j) * Azz(i,j,k) + &
        (p_gupxy(i,j) * p_gupyz(i,j)       + p_gupyy(i,j) * p_gupxz(i,j))* Axy(i,j,k)                       + &
        (p_gupxy(i,j) * p_gupzz(i,j)       + p_gupyz(i,j) * p_gupxz(i,j))* Axz(i,j,k)                       + &
        (p_gupyy(i,j) * p_gupzz(i,j)       + p_gupyz(i,j) * p_gupyz(i,j))* Ayz(i,j,k)

      end do
    end do
  call fderivs_plane(Lap, p_Lapx, p_Lapy, p_Lapz, SYM, SYM, SYM)
  call fderivs_plane(trK, p_Kx, p_Ky, p_Kz, SYM, SYM, SYM)
  do j = 1, ex(2)
  do i = 1, ex(1)
  alpn1 = Lap(i,j,k) + ONE
  chin1 = chi(i,j,k) + ONE
  Gamx_rhs(i,j,k) = - TWO * (   p_Lapx(i,j) * Rxx(i,j,k) +   p_Lapy(i,j) * Rxy(i,j,k) +   p_Lapz(i,j) * Rxz(i,j,k) ) + &
  TWO * alpn1 * (                                                &
  -F3o2/chin1 * (   p_chix(i,j) * Rxx(i,j,k) +   p_chiy(i,j) * Rxy(i,j,k) +   p_chiz(i,j) * Rxz(i,j,k) ) - &
  p_gupxx(i,j) * (   F2o3 * p_Kx(i,j)  +  EIGHT * PI * Sx(i,j,k)            ) - &
  p_gupxy(i,j) * (   F2o3 * p_Ky(i,j)  +  EIGHT * PI * Sy(i,j,k)            ) - &
  p_gupxz(i,j) * (   F2o3 * p_Kz(i,j)  +  EIGHT * PI * Sz(i,j,k)            ) + &
  Gamxxx(i,j,k) * Rxx(i,j,k) + Gamxyy(i,j,k) * Ryy(i,j,k) + Gamxzz(i,j,k) * Rzz(i,j,k)   + &
  TWO * ( Gamxxy(i,j,k) * Rxy(i,j,k) + Gamxxz(i,j,k) * Rxz(i,j,k) + Gamxyz(i,j,k) * Ryz(i,j,k) ) )

  Gamy_rhs(i,j,k) = - TWO * (   p_Lapx(i,j) * Rxy(i,j,k) +   p_Lapy(i,j) * Ryy(i,j,k) +   p_Lapz(i,j) * Ryz(i,j,k) ) + &
  TWO * alpn1 * (                                                &
  -F3o2/chin1 * (   p_chix(i,j) * Rxy(i,j,k) +  p_chiy(i,j) * Ryy(i,j,k) +    p_chiz(i,j) * Ryz(i,j,k) ) - &
  p_gupxy(i,j) * (   F2o3 * p_Kx(i,j)  +  EIGHT * PI * Sx(i,j,k)            ) - &
  p_gupyy(i,j) * (   F2o3 * p_Ky(i,j)  +  EIGHT * PI * Sy(i,j,k)            ) - &
  p_gupyz(i,j) * (   F2o3 * p_Kz(i,j)  +  EIGHT * PI * Sz(i,j,k)            ) + &
  Gamyxx(i,j,k) * Rxx(i,j,k) + Gamyyy(i,j,k) * Ryy(i,j,k) + Gamyzz(i,j,k) * Rzz(i,j,k)   + &
  TWO * ( Gamyxy(i,j,k) * Rxy(i,j,k) + Gamyxz(i,j,k) * Rxz(i,j,k) + Gamyyz(i,j,k) * Ryz(i,j,k) ) )

  Gamz_rhs(i,j,k) = - TWO * (   p_Lapx(i,j) * Rxz(i,j,k) +   p_Lapy(i,j) * Ryz(i,j,k) +   p_Lapz(i,j) * Rzz(i,j,k) ) + &
  TWO * alpn1 * (                                                &
  -F3o2/chin1 * (   p_chix(i,j) * Rxz(i,j,k) +  p_chiy(i,j) * Ryz(i,j,k) +    p_chiz(i,j) * Rzz(i,j,k) ) - &
  p_gupxz(i,j) * (   F2o3 * p_Kx(i,j)  +  EIGHT * PI * Sx(i,j,k)            ) - &
  p_gupyz(i,j) * (   F2o3 * p_Ky(i,j)  +  EIGHT * PI * Sy(i,j,k)            ) - &
  p_gupzz(i,j) * (   F2o3 * p_Kz(i,j)  +  EIGHT * PI * Sz(i,j,k)            ) + &
  Gamzxx(i,j,k) * Rxx(i,j,k) + Gamzyy(i,j,k) * Ryy(i,j,k) + Gamzzz(i,j,k) * Rzz(i,j,k)   + &
  TWO * ( Gamzxy(i,j,k) * Rxy(i,j,k) + Gamzxz(i,j,k) * Rxz(i,j,k) + Gamzyz(i,j,k) * Ryz(i,j,k) ) )
  end do
  end do
  call fdderivs_plane(betax, p_gxxx, p_gxyx, p_gxzx, p_gyyx, p_gyzx, p_gzzx, ANTI, SYM, SYM)
  call fdderivs_plane(betay, p_gxxy, p_gxyy, p_gxzy, p_gyyy, p_gyzy, p_gzzy, SYM, ANTI, SYM)
  call fdderivs_plane(betaz, p_gxxz, p_gxyz, p_gxzz, p_gyyz, p_gyzz, p_gzzz, SYM, SYM, ANTI)
  do j = 1, ex(2)
  do i = 1, ex(1)
  p_Gamxa(i,j) =       p_gupxx(i,j) * Gamxxx(i,j,k) + p_gupyy(i,j) * Gamxyy(i,j,k) + p_gupzz(i,j) * Gamxzz(i,j,k) + &
  TWO*( p_gupxy(i,j) * Gamxxy(i,j,k) + p_gupxz(i,j) * Gamxxz(i,j,k) + p_gupyz(i,j) * Gamxyz(i,j,k) )
  p_Gamya(i,j) =       p_gupxx(i,j) * Gamyxx(i,j,k) + p_gupyy(i,j) * Gamyyy(i,j,k) + p_gupzz(i,j) * Gamyzz(i,j,k) + &
  TWO*( p_gupxy(i,j) * Gamyxy(i,j,k) + p_gupxz(i,j) * Gamyxz(i,j,k) + p_gupyz(i,j) * Gamyyz(i,j,k) )
  p_Gamza(i,j) =       p_gupxx(i,j) * Gamzxx(i,j,k) + p_gupyy(i,j) * Gamzyy(i,j,k) + p_gupzz(i,j) * Gamzzz(i,j,k) + &
  TWO*( p_gupxy(i,j) * Gamzxy(i,j,k) + p_gupxz(i,j) * Gamzxz(i,j,k) + p_gupyz(i,j) * Gamzyz(i,j,k) )
  end do
  end do
  call fderivs_plane(Gamx, p_Gamxx, p_Gamxy, p_Gamxz, ANTI, SYM, SYM)
  call fderivs_plane(Gamy, p_Gamyx, p_Gamyy, p_Gamyz, SYM, ANTI, SYM)
  call fderivs_plane(Gamz, p_Gamzx, p_Gamzy, p_Gamzz, SYM, SYM, ANTI)
  do j = 1, ex(2)
  do i = 1, ex(1)
  div_beta = p_betaxx(i,j) + p_betayy(i,j) + p_betazz(i,j)
  s_fxx = p_gxxx(i,j) + p_gxyy(i,j) + p_gxzz(i,j)
  s_fxy = p_gxyx(i,j) + p_gyyy(i,j) + p_gyzz(i,j)
  s_fxz = p_gxzx(i,j) + p_gyzy(i,j) + p_gzzz(i,j)
  Gamx_rhs(i,j,k) = Gamx_rhs(i,j,k) +  F2o3 *  p_Gamxa(i,j) * div_beta        - &
  p_Gamxa(i,j) * p_betaxx(i,j) - p_Gamya(i,j) * p_betaxy(i,j) - p_Gamza(i,j) * p_betaxz(i,j)  + &
  F1o3 * (p_gupxx(i,j) * s_fxx    + p_gupxy(i,j) * s_fxy    + p_gupxz(i,j) * s_fxz    ) + &
  p_gupxx(i,j) * p_gxxx(i,j)   + p_gupyy(i,j) * p_gyyx(i,j)   + p_gupzz(i,j) * p_gzzx(i,j)    + &
  TWO * (p_gupxy(i,j) * p_gxyx(i,j)   + p_gupxz(i,j) * p_gxzx(i,j)   + p_gupyz(i,j) * p_gyzx(i,j)  )

  Gamy_rhs(i,j,k) = Gamy_rhs(i,j,k) +  F2o3 *  p_Gamya(i,j) * div_beta        - &
  p_Gamxa(i,j) * p_betayx(i,j) - p_Gamya(i,j) * p_betayy(i,j) - p_Gamza(i,j) * p_betayz(i,j)  + &
  F1o3 * (p_gupxy(i,j) * s_fxx    + p_gupyy(i,j) * s_fxy    + p_gupyz(i,j) * s_fxz    ) + &
  p_gupxx(i,j) * p_gxxy(i,j)   + p_gupyy(i,j) * p_gyyy(i,j)   + p_gupzz(i,j) * p_gzzy(i,j)    + &
  TWO * (p_gupxy(i,j) * p_gxyy(i,j)   + p_gupxz(i,j) * p_gxzy(i,j)   + p_gupyz(i,j) * p_gyzy(i,j)  )

  Gamz_rhs(i,j,k) = Gamz_rhs(i,j,k) +  F2o3 *  p_Gamza(i,j) * div_beta        - &
  p_Gamxa(i,j) * p_betazx(i,j) - p_Gamya(i,j) * p_betazy(i,j) - p_Gamza(i,j) * p_betazz(i,j)  + &
  F1o3 * (p_gupxz(i,j) * s_fxx    + p_gupyz(i,j) * s_fxy    + p_gupzz(i,j) * s_fxz    ) + &
  p_gupxx(i,j) * p_gxxz(i,j)   + p_gupyy(i,j) * p_gyyz(i,j)   + p_gupzz(i,j) * p_gzzz(i,j)    + &
  TWO * (p_gupxy(i,j) * p_gxyz(i,j)   + p_gupxz(i,j) * p_gxzz(i,j)   + p_gupyz(i,j) * p_gyzz(i,j)  )    !rhs for Gam^i
  end do
  end do
    do j = 1, ex(2)
      do i = 1, ex(1)
        p_gxxx(i,j) = gxx(i,j,k) * Gamxxx(i,j,k) + gxy(i,j,k) * Gamyxx(i,j,k) + gxz(i,j,k) * Gamzxx(i,j,k)
        p_gxyx(i,j) = gxx(i,j,k) * Gamxxy(i,j,k) + gxy(i,j,k) * Gamyxy(i,j,k) + gxz(i,j,k) * Gamzxy(i,j,k)
        p_gxzx(i,j) = gxx(i,j,k) * Gamxxz(i,j,k) + gxy(i,j,k) * Gamyxz(i,j,k) + gxz(i,j,k) * Gamzxz(i,j,k)
        p_gyyx(i,j) = gxx(i,j,k) * Gamxyy(i,j,k) + gxy(i,j,k) * Gamyyy(i,j,k) + gxz(i,j,k) * Gamzyy(i,j,k)
        p_gyzx(i,j) = gxx(i,j,k) * Gamxyz(i,j,k) + gxy(i,j,k) * Gamyyz(i,j,k) + gxz(i,j,k) * Gamzyz(i,j,k)
        p_gzzx(i,j) = gxx(i,j,k) * Gamxzz(i,j,k) + gxy(i,j,k) * Gamyzz(i,j,k) + gxz(i,j,k) * Gamzzz(i,j,k)

        p_gxxy(i,j) = gxy(i,j,k) * Gamxxx(i,j,k) + gyy(i,j,k) * Gamyxx(i,j,k) + gyz(i,j,k) * Gamzxx(i,j,k)
        p_gxyy(i,j) = gxy(i,j,k) * Gamxxy(i,j,k) + gyy(i,j,k) * Gamyxy(i,j,k) + gyz(i,j,k) * Gamzxy(i,j,k)
        p_gxzy(i,j) = gxy(i,j,k) * Gamxxz(i,j,k) + gyy(i,j,k) * Gamyxz(i,j,k) + gyz(i,j,k) * Gamzxz(i,j,k)
        p_gyyy(i,j) = gxy(i,j,k) * Gamxyy(i,j,k) + gyy(i,j,k) * Gamyyy(i,j,k) + gyz(i,j,k) * Gamzyy(i,j,k)
        p_gyzy(i,j) = gxy(i,j,k) * Gamxyz(i,j,k) + gyy(i,j,k) * Gamyyz(i,j,k) + gyz(i,j,k) * Gamzyz(i,j,k)
        p_gzzy(i,j) = gxy(i,j,k) * Gamxzz(i,j,k) + gyy(i,j,k) * Gamyzz(i,j,k) + gyz(i,j,k) * Gamzzz(i,j,k)

        p_gxxz(i,j) = gxz(i,j,k) * Gamxxx(i,j,k) + gyz(i,j,k) * Gamyxx(i,j,k) + gzz(i,j,k) * Gamzxx(i,j,k)
        p_gxyz(i,j) = gxz(i,j,k) * Gamxxy(i,j,k) + gyz(i,j,k) * Gamyxy(i,j,k) + gzz(i,j,k) * Gamzxy(i,j,k)
        p_gxzz(i,j) = gxz(i,j,k) * Gamxxz(i,j,k) + gyz(i,j,k) * Gamyxz(i,j,k) + gzz(i,j,k) * Gamzxz(i,j,k)
        p_gyyz(i,j) = gxz(i,j,k) * Gamxyy(i,j,k) + gyz(i,j,k) * Gamyyy(i,j,k) + gzz(i,j,k) * Gamzyy(i,j,k)
        p_gyzz(i,j) = gxz(i,j,k) * Gamxyz(i,j,k) + gyz(i,j,k) * Gamyyz(i,j,k) + gzz(i,j,k) * Gamzyz(i,j,k)
        p_gzzz(i,j) = gxz(i,j,k) * Gamxzz(i,j,k) + gyz(i,j,k) * Gamyzz(i,j,k) + gzz(i,j,k) * Gamzzz(i,j,k)

      end do
    end do
  call fdderivs_plane(dxx, p_fxx, p_fxy, p_fxz, p_fyy, p_fyz, p_fzz, SYM, SYM, SYM)
    do j = 1, ex(2)
      do i = 1, ex(1)
        Rxx(i,j,k) =   p_gupxx(i,j) * p_fxx(i,j) + p_gupyy(i,j) * p_fyy(i,j) + p_gupzz(i,j) * p_fzz(i,j) + &
        ( p_gupxy(i,j) * p_fxy(i,j) + p_gupxz(i,j) * p_fxz(i,j) + p_gupyz(i,j) * p_fyz(i,j) ) * TWO
      end do
    end do
  call fdderivs_plane(dyy, p_fxx, p_fxy, p_fxz, p_fyy, p_fyz, p_fzz, SYM, SYM, SYM)
    do j = 1, ex(2)
      do i = 1, ex(1)
        Ryy(i,j,k) =   p_gupxx(i,j) * p_fxx(i,j) + p_gupyy(i,j) * p_fyy(i,j) + p_gupzz(i,j) * p_fzz(i,j) + &
        ( p_gupxy(i,j) * p_fxy(i,j) + p_gupxz(i,j) * p_fxz(i,j) + p_gupyz(i,j) * p_fyz(i,j) ) * TWO
      end do
    end do
  call fdderivs_plane(dzz, p_fxx, p_fxy, p_fxz, p_fyy, p_fyz, p_fzz, SYM, SYM, SYM)
    do j = 1, ex(2)
      do i = 1, ex(1)
        Rzz(i,j,k) =   p_gupxx(i,j) * p_fxx(i,j) + p_gupyy(i,j) * p_fyy(i,j) + p_gupzz(i,j) * p_fzz(i,j) + &
        ( p_gupxy(i,j) * p_fxy(i,j) + p_gupxz(i,j) * p_fxz(i,j) + p_gupyz(i,j) * p_fyz(i,j) ) * TWO
      end do
    end do
  call fdderivs_plane(gxy, p_fxx, p_fxy, p_fxz, p_fyy, p_fyz, p_fzz, ANTI, ANTI, SYM)
    do j = 1, ex(2)
      do i = 1, ex(1)
        Rxy(i,j,k) =   p_gupxx(i,j) * p_fxx(i,j) + p_gupyy(i,j) * p_fyy(i,j) + p_gupzz(i,j) * p_fzz(i,j) + &
        ( p_gupxy(i,j) * p_fxy(i,j) + p_gupxz(i,j) * p_fxz(i,j) + p_gupyz(i,j) * p_fyz(i,j) ) * TWO
      end do
    end do
  call fdderivs_plane(gxz, p_fxx, p_fxy, p_fxz, p_fyy, p_fyz, p_fzz, ANTI, SYM, ANTI)
    do j = 1, ex(2)
      do i = 1, ex(1)
        Rxz(i,j,k) =   p_gupxx(i,j) * p_fxx(i,j) + p_gupyy(i,j) * p_fyy(i,j) + p_gupzz(i,j) * p_fzz(i,j) + &
        ( p_gupxy(i,j) * p_fxy(i,j) + p_gupxz(i,j) * p_fxz(i,j) + p_gupyz(i,j) * p_fyz(i,j) ) * TWO
      end do
    end do
  call fdderivs_plane(gyz, p_fxx, p_fxy, p_fxz, p_fyy, p_fyz, p_fzz, SYM, ANTI, ANTI)
    do j = 1, ex(2)
      do i = 1, ex(1)
        Ryz(i,j,k) =   p_gupxx(i,j) * p_fxx(i,j) + p_gupyy(i,j) * p_fyy(i,j) + p_gupzz(i,j) * p_fzz(i,j) + &
        ( p_gupxy(i,j) * p_fxy(i,j) + p_gupxz(i,j) * p_fxz(i,j) + p_gupyz(i,j) * p_fyz(i,j) ) * TWO
      end do
    end do
    do j = 1, ex(2)
      do i = 1, ex(1)
        Rxx(i,j,k) =     - HALF * Rxx(i,j,k)                                   + &
        gxx(i,j,k) * p_Gamxx(i,j)+ gxy(i,j,k) * p_Gamyx(i,j)   +    gxz(i,j,k) * p_Gamzx(i,j) + &
        p_Gamxa(i,j) * p_gxxx(i,j) +  p_Gamya(i,j) * p_gxyx(i,j) +  p_Gamza(i,j) * p_gxzx(i,j)  + &
        p_gupxx(i,j) *(                                                  &
        TWO*(Gamxxx(i,j,k) * p_gxxx(i,j) + Gamyxx(i,j,k) * p_gxyx(i,j) + Gamzxx(i,j,k) * p_gxzx(i,j)) + &
        Gamxxx(i,j,k) * p_gxxx(i,j) + Gamyxx(i,j,k) * p_gxxy(i,j) + Gamzxx(i,j,k) * p_gxxz(i,j) )+ &
        p_gupxy(i,j) *(                                                  &
        TWO*(Gamxxx(i,j,k) * p_gxyx(i,j) + Gamyxx(i,j,k) * p_gyyx(i,j) + Gamzxx(i,j,k) * p_gyzx(i,j)  + &
        Gamxxy(i,j,k) * p_gxxx(i,j) + Gamyxy(i,j,k) * p_gxyx(i,j) + Gamzxy(i,j,k) * p_gxzx(i,j)) + &
        Gamxxy(i,j,k) * p_gxxx(i,j) + Gamyxy(i,j,k) * p_gxxy(i,j) + Gamzxy(i,j,k) * p_gxxz(i,j)  + &
        Gamxxx(i,j,k) * p_gxyx(i,j) + Gamyxx(i,j,k) * p_gxyy(i,j) + Gamzxx(i,j,k) * p_gxyz(i,j) )+ &
        p_gupxz(i,j) *(                                                  &
        TWO*(Gamxxx(i,j,k) * p_gxzx(i,j) + Gamyxx(i,j,k) * p_gyzx(i,j) + Gamzxx(i,j,k) * p_gzzx(i,j)  + &
        Gamxxz(i,j,k) * p_gxxx(i,j) + Gamyxz(i,j,k) * p_gxyx(i,j) + Gamzxz(i,j,k) * p_gxzx(i,j)) + &
        Gamxxz(i,j,k) * p_gxxx(i,j) + Gamyxz(i,j,k) * p_gxxy(i,j) + Gamzxz(i,j,k) * p_gxxz(i,j)  + &
        Gamxxx(i,j,k) * p_gxzx(i,j) + Gamyxx(i,j,k) * p_gxzy(i,j) + Gamzxx(i,j,k) * p_gxzz(i,j) )+ &
        p_gupyy(i,j) *(                                                  &
        TWO*(Gamxxy(i,j,k) * p_gxyx(i,j) + Gamyxy(i,j,k) * p_gyyx(i,j) + Gamzxy(i,j,k) * p_gyzx(i,j)) + &
        Gamxxy(i,j,k) * p_gxyx(i,j) + Gamyxy(i,j,k) * p_gxyy(i,j) + Gamzxy(i,j,k) * p_gxyz(i,j) )+ &
        p_gupyz(i,j) *(                                                  &
        TWO*(Gamxxy(i,j,k) * p_gxzx(i,j) + Gamyxy(i,j,k) * p_gyzx(i,j) + Gamzxy(i,j,k) * p_gzzx(i,j)  + &
        Gamxxz(i,j,k) * p_gxyx(i,j) + Gamyxz(i,j,k) * p_gyyx(i,j) + Gamzxz(i,j,k) * p_gyzx(i,j)) + &
        Gamxxz(i,j,k) * p_gxyx(i,j) + Gamyxz(i,j,k) * p_gxyy(i,j) + Gamzxz(i,j,k) * p_gxyz(i,j)  + &
        Gamxxy(i,j,k) * p_gxzx(i,j) + Gamyxy(i,j,k) * p_gxzy(i,j) + Gamzxy(i,j,k) * p_gxzz(i,j) )+ &
        p_gupzz(i,j) *(                                                  &
        TWO*(Gamxxz(i,j,k) * p_gxzx(i,j) + Gamyxz(i,j,k) * p_gyzx(i,j) + Gamzxz(i,j,k) * p_gzzx(i,j)) + &
        Gamxxz(i,j,k) * p_gxzx(i,j) + Gamyxz(i,j,k) * p_gxzy(i,j) + Gamzxz(i,j,k) * p_gxzz(i,j) )

        Ryy(i,j,k) =     - HALF * Ryy(i,j,k)                                   + &
        gxy(i,j,k) * p_Gamxy(i,j)+  gyy(i,j,k) * p_Gamyy(i,j)  +  gyz(i,j,k) * p_Gamzy(i,j)   + &
        p_Gamxa(i,j) * p_gxyy(i,j) +  p_Gamya(i,j) * p_gyyy(i,j) +  p_Gamza(i,j) * p_gyzy(i,j)  + &
        p_gupxx(i,j) *(                                                  &
        TWO*(Gamxxy(i,j,k) * p_gxxy(i,j) + Gamyxy(i,j,k) * p_gxyy(i,j) + Gamzxy(i,j,k) * p_gxzy(i,j)) + &
        Gamxxy(i,j,k) * p_gxyx(i,j) + Gamyxy(i,j,k) * p_gxyy(i,j) + Gamzxy(i,j,k) * p_gxyz(i,j) )+ &
        p_gupxy(i,j) *(                                                  &
        TWO*(Gamxxy(i,j,k) * p_gxyy(i,j) + Gamyxy(i,j,k) * p_gyyy(i,j) + Gamzxy(i,j,k) * p_gyzy(i,j)  + &
        Gamxyy(i,j,k) * p_gxxy(i,j) + Gamyyy(i,j,k) * p_gxyy(i,j) + Gamzyy(i,j,k) * p_gxzy(i,j)) + &
        Gamxyy(i,j,k) * p_gxyx(i,j) + Gamyyy(i,j,k) * p_gxyy(i,j) + Gamzyy(i,j,k) * p_gxyz(i,j)  + &
        Gamxxy(i,j,k) * p_gyyx(i,j) + Gamyxy(i,j,k) * p_gyyy(i,j) + Gamzxy(i,j,k) * p_gyyz(i,j) )+ &
        p_gupxz(i,j) *(                                                  &
        TWO*(Gamxxy(i,j,k) * p_gxzy(i,j) + Gamyxy(i,j,k) * p_gyzy(i,j) + Gamzxy(i,j,k) * p_gzzy(i,j)  + &
        Gamxyz(i,j,k) * p_gxxy(i,j) + Gamyyz(i,j,k) * p_gxyy(i,j) + Gamzyz(i,j,k) * p_gxzy(i,j)) + &
        Gamxyz(i,j,k) * p_gxyx(i,j) + Gamyyz(i,j,k) * p_gxyy(i,j) + Gamzyz(i,j,k) * p_gxyz(i,j)  + &
        Gamxxy(i,j,k) * p_gyzx(i,j) + Gamyxy(i,j,k) * p_gyzy(i,j) + Gamzxy(i,j,k) * p_gyzz(i,j) )+ &
        p_gupyy(i,j) *(                                                  &
        TWO*(Gamxyy(i,j,k) * p_gxyy(i,j) + Gamyyy(i,j,k) * p_gyyy(i,j) + Gamzyy(i,j,k) * p_gyzy(i,j)) + &
        Gamxyy(i,j,k) * p_gyyx(i,j) + Gamyyy(i,j,k) * p_gyyy(i,j) + Gamzyy(i,j,k) * p_gyyz(i,j) )+ &
        p_gupyz(i,j) *(                                                  &
        TWO*(Gamxyy(i,j,k) * p_gxzy(i,j) + Gamyyy(i,j,k) * p_gyzy(i,j) + Gamzyy(i,j,k) * p_gzzy(i,j)  + &
        Gamxyz(i,j,k) * p_gxyy(i,j) + Gamyyz(i,j,k) * p_gyyy(i,j) + Gamzyz(i,j,k) * p_gyzy(i,j)) + &
        Gamxyz(i,j,k) * p_gyyx(i,j) + Gamyyz(i,j,k) * p_gyyy(i,j) + Gamzyz(i,j,k) * p_gyyz(i,j)  + &
        Gamxyy(i,j,k) * p_gyzx(i,j) + Gamyyy(i,j,k) * p_gyzy(i,j) + Gamzyy(i,j,k) * p_gyzz(i,j) )+ &
        p_gupzz(i,j) *(                                                  &
        TWO*(Gamxyz(i,j,k) * p_gxzy(i,j) + Gamyyz(i,j,k) * p_gyzy(i,j) + Gamzyz(i,j,k) * p_gzzy(i,j)) + &
        Gamxyz(i,j,k) * p_gyzx(i,j) + Gamyyz(i,j,k) * p_gyzy(i,j) + Gamzyz(i,j,k) * p_gyzz(i,j) )

        Rzz(i,j,k) =     - HALF * Rzz(i,j,k)                                   + &
        gxz(i,j,k) * p_Gamxz(i,j)+ gyz(i,j,k) * p_Gamyz(i,j)  +    gzz(i,j,k) * p_Gamzz(i,j)  + &
        p_Gamxa(i,j) * p_gxzz(i,j) +  p_Gamya(i,j) * p_gyzz(i,j) +  p_Gamza(i,j) * p_gzzz(i,j)  + &
        p_gupxx(i,j) *(                                                  &
        TWO*(Gamxxz(i,j,k) * p_gxxz(i,j) + Gamyxz(i,j,k) * p_gxyz(i,j) + Gamzxz(i,j,k) * p_gxzz(i,j)) + &
        Gamxxz(i,j,k) * p_gxzx(i,j) + Gamyxz(i,j,k) * p_gxzy(i,j) + Gamzxz(i,j,k) * p_gxzz(i,j) )+ &
        p_gupxy(i,j) *(                                                  &
        TWO*(Gamxxz(i,j,k) * p_gxyz(i,j) + Gamyxz(i,j,k) * p_gyyz(i,j) + Gamzxz(i,j,k) * p_gyzz(i,j)  + &
        Gamxyz(i,j,k) * p_gxxz(i,j) + Gamyyz(i,j,k) * p_gxyz(i,j) + Gamzyz(i,j,k) * p_gxzz(i,j)) + &
        Gamxyz(i,j,k) * p_gxzx(i,j) + Gamyyz(i,j,k) * p_gxzy(i,j) + Gamzyz(i,j,k) * p_gxzz(i,j)  + &
        Gamxxz(i,j,k) * p_gyzx(i,j) + Gamyxz(i,j,k) * p_gyzy(i,j) + Gamzxz(i,j,k) * p_gyzz(i,j) )+ &
        p_gupxz(i,j) *(                                                  &
        TWO*(Gamxxz(i,j,k) * p_gxzz(i,j) + Gamyxz(i,j,k) * p_gyzz(i,j) + Gamzxz(i,j,k) * p_gzzz(i,j)  + &
        Gamxzz(i,j,k) * p_gxxz(i,j) + Gamyzz(i,j,k) * p_gxyz(i,j) + Gamzzz(i,j,k) * p_gxzz(i,j)) + &
        Gamxzz(i,j,k) * p_gxzx(i,j) + Gamyzz(i,j,k) * p_gxzy(i,j) + Gamzzz(i,j,k) * p_gxzz(i,j)  + &
        Gamxxz(i,j,k) * p_gzzx(i,j) + Gamyxz(i,j,k) * p_gzzy(i,j) + Gamzxz(i,j,k) * p_gzzz(i,j) )+ &
        p_gupyy(i,j) *(                                                  &
        TWO*(Gamxyz(i,j,k) * p_gxyz(i,j) + Gamyyz(i,j,k) * p_gyyz(i,j) + Gamzyz(i,j,k) * p_gyzz(i,j)) + &
        Gamxyz(i,j,k) * p_gyzx(i,j) + Gamyyz(i,j,k) * p_gyzy(i,j) + Gamzyz(i,j,k) * p_gyzz(i,j) )+ &
        p_gupyz(i,j) *(                                                  &
        TWO*(Gamxyz(i,j,k) * p_gxzz(i,j) + Gamyyz(i,j,k) * p_gyzz(i,j) + Gamzyz(i,j,k) * p_gzzz(i,j)  + &
        Gamxzz(i,j,k) * p_gxyz(i,j) + Gamyzz(i,j,k) * p_gyyz(i,j) + Gamzzz(i,j,k) * p_gyzz(i,j)) + &
        Gamxzz(i,j,k) * p_gyzx(i,j) + Gamyzz(i,j,k) * p_gyzy(i,j) + Gamzzz(i,j,k) * p_gyzz(i,j)  + &
        Gamxyz(i,j,k) * p_gzzx(i,j) + Gamyyz(i,j,k) * p_gzzy(i,j) + Gamzyz(i,j,k) * p_gzzz(i,j) )+ &
        p_gupzz(i,j) *(                                                  &
        TWO*(Gamxzz(i,j,k) * p_gxzz(i,j) + Gamyzz(i,j,k) * p_gyzz(i,j) + Gamzzz(i,j,k) * p_gzzz(i,j)) + &
        Gamxzz(i,j,k) * p_gzzx(i,j) + Gamyzz(i,j,k) * p_gzzy(i,j) + Gamzzz(i,j,k) * p_gzzz(i,j) )

        Rxy(i,j,k) = HALF*(     - Rxy(i,j,k)                                   + &
        gxx(i,j,k) * p_Gamxy(i,j) +    gxy(i,j,k) * p_Gamyy(i,j) + gxz(i,j,k) * p_Gamzy(i,j)  + &
        gxy(i,j,k) * p_Gamxx(i,j) +    gyy(i,j,k) * p_Gamyx(i,j) + gyz(i,j,k) * p_Gamzx(i,j)  + &
        p_Gamxa(i,j) * p_gxyx(i,j) +  p_Gamya(i,j) * p_gyyx(i,j) +  p_Gamza(i,j) * p_gyzx(i,j)  + &
        p_Gamxa(i,j) * p_gxxy(i,j) +  p_Gamya(i,j) * p_gxyy(i,j) +  p_Gamza(i,j) * p_gxzy(i,j) )+ &
        p_gupxx(i,j) *(                                                  &
        Gamxxx(i,j,k) * p_gxxy(i,j) + Gamyxx(i,j,k) * p_gxyy(i,j) + Gamzxx(i,j,k) * p_gxzy(i,j)  + &
        Gamxxy(i,j,k) * p_gxxx(i,j) + Gamyxy(i,j,k) * p_gxyx(i,j) + Gamzxy(i,j,k) * p_gxzx(i,j)  + &
        Gamxxx(i,j,k) * p_gxyx(i,j) + Gamyxx(i,j,k) * p_gxyy(i,j) + Gamzxx(i,j,k) * p_gxyz(i,j) )+ &
        p_gupxy(i,j) *(                                                  &
        Gamxxx(i,j,k) * p_gxyy(i,j) + Gamyxx(i,j,k) * p_gyyy(i,j) + Gamzxx(i,j,k) * p_gyzy(i,j)  + &
        Gamxxy(i,j,k) * p_gxyx(i,j) + Gamyxy(i,j,k) * p_gyyx(i,j) + Gamzxy(i,j,k) * p_gyzx(i,j)  + &
        Gamxxy(i,j,k) * p_gxyx(i,j) + Gamyxy(i,j,k) * p_gxyy(i,j) + Gamzxy(i,j,k) * p_gxyz(i,j)  + &
        Gamxxy(i,j,k) * p_gxxy(i,j) + Gamyxy(i,j,k) * p_gxyy(i,j) + Gamzxy(i,j,k) * p_gxzy(i,j)  + &
        Gamxyy(i,j,k) * p_gxxx(i,j) + Gamyyy(i,j,k) * p_gxyx(i,j) + Gamzyy(i,j,k) * p_gxzx(i,j)  + &
        Gamxxx(i,j,k) * p_gyyx(i,j) + Gamyxx(i,j,k) * p_gyyy(i,j) + Gamzxx(i,j,k) * p_gyyz(i,j) )+ &
        p_gupxz(i,j) *(                                                  &
        Gamxxx(i,j,k) * p_gxzy(i,j) + Gamyxx(i,j,k) * p_gyzy(i,j) + Gamzxx(i,j,k) * p_gzzy(i,j)  + &
        Gamxxy(i,j,k) * p_gxzx(i,j) + Gamyxy(i,j,k) * p_gyzx(i,j) + Gamzxy(i,j,k) * p_gzzx(i,j)  + &
        Gamxxz(i,j,k) * p_gxyx(i,j) + Gamyxz(i,j,k) * p_gxyy(i,j) + Gamzxz(i,j,k) * p_gxyz(i,j)  + &
        Gamxxz(i,j,k) * p_gxxy(i,j) + Gamyxz(i,j,k) * p_gxyy(i,j) + Gamzxz(i,j,k) * p_gxzy(i,j)  + &
        Gamxyz(i,j,k) * p_gxxx(i,j) + Gamyyz(i,j,k) * p_gxyx(i,j) + Gamzyz(i,j,k) * p_gxzx(i,j)  + &
        Gamxxx(i,j,k) * p_gyzx(i,j) + Gamyxx(i,j,k) * p_gyzy(i,j) + Gamzxx(i,j,k) * p_gyzz(i,j) )+ &
        p_gupyy(i,j) *(                                                  &
        Gamxxy(i,j,k) * p_gxyy(i,j) + Gamyxy(i,j,k) * p_gyyy(i,j) + Gamzxy(i,j,k) * p_gyzy(i,j)  + &
        Gamxyy(i,j,k) * p_gxyx(i,j) + Gamyyy(i,j,k) * p_gyyx(i,j) + Gamzyy(i,j,k) * p_gyzx(i,j)  + &
        Gamxxy(i,j,k) * p_gyyx(i,j) + Gamyxy(i,j,k) * p_gyyy(i,j) + Gamzxy(i,j,k) * p_gyyz(i,j) )+ &
        p_gupyz(i,j) *(                                                  &
        Gamxxy(i,j,k) * p_gxzy(i,j) + Gamyxy(i,j,k) * p_gyzy(i,j) + Gamzxy(i,j,k) * p_gzzy(i,j)  + &
        Gamxyy(i,j,k) * p_gxzx(i,j) + Gamyyy(i,j,k) * p_gyzx(i,j) + Gamzyy(i,j,k) * p_gzzx(i,j)  + &
        Gamxxz(i,j,k) * p_gyyx(i,j) + Gamyxz(i,j,k) * p_gyyy(i,j) + Gamzxz(i,j,k) * p_gyyz(i,j)  + &
        Gamxxz(i,j,k) * p_gxyy(i,j) + Gamyxz(i,j,k) * p_gyyy(i,j) + Gamzxz(i,j,k) * p_gyzy(i,j)  + &
        Gamxyz(i,j,k) * p_gxyx(i,j) + Gamyyz(i,j,k) * p_gyyx(i,j) + Gamzyz(i,j,k) * p_gyzx(i,j)  + &
        Gamxxy(i,j,k) * p_gyzx(i,j) + Gamyxy(i,j,k) * p_gyzy(i,j) + Gamzxy(i,j,k) * p_gyzz(i,j) )+ &
        p_gupzz(i,j) *(                                                  &
        Gamxxz(i,j,k) * p_gxzy(i,j) + Gamyxz(i,j,k) * p_gyzy(i,j) + Gamzxz(i,j,k) * p_gzzy(i,j)  + &
        Gamxyz(i,j,k) * p_gxzx(i,j) + Gamyyz(i,j,k) * p_gyzx(i,j) + Gamzyz(i,j,k) * p_gzzx(i,j)  + &
        Gamxxz(i,j,k) * p_gyzx(i,j) + Gamyxz(i,j,k) * p_gyzy(i,j) + Gamzxz(i,j,k) * p_gyzz(i,j) )

        Rxz(i,j,k) = HALF*(     - Rxz(i,j,k)                                   + &
        gxx(i,j,k) * p_Gamxz(i,j) +  gxy(i,j,k) * p_Gamyz(i,j) + gxz(i,j,k) * p_Gamzz(i,j)    + &
        gxz(i,j,k) * p_Gamxx(i,j) +  gyz(i,j,k) * p_Gamyx(i,j) + gzz(i,j,k) * p_Gamzx(i,j)    + &
        p_Gamxa(i,j) * p_gxzx(i,j) +  p_Gamya(i,j) * p_gyzx(i,j) +  p_Gamza(i,j) * p_gzzx(i,j)  + &
        p_Gamxa(i,j) * p_gxxz(i,j) +  p_Gamya(i,j) * p_gxyz(i,j) +  p_Gamza(i,j) * p_gxzz(i,j) )+ &
        p_gupxx(i,j) *(                                                  &
        Gamxxx(i,j,k) * p_gxxz(i,j) + Gamyxx(i,j,k) * p_gxyz(i,j) + Gamzxx(i,j,k) * p_gxzz(i,j)  + &
        Gamxxz(i,j,k) * p_gxxx(i,j) + Gamyxz(i,j,k) * p_gxyx(i,j) + Gamzxz(i,j,k) * p_gxzx(i,j)  + &
        Gamxxx(i,j,k) * p_gxzx(i,j) + Gamyxx(i,j,k) * p_gxzy(i,j) + Gamzxx(i,j,k) * p_gxzz(i,j) )+ &
        p_gupxy(i,j) *(                                                  &
        Gamxxx(i,j,k) * p_gxyz(i,j) + Gamyxx(i,j,k) * p_gyyz(i,j) + Gamzxx(i,j,k) * p_gyzz(i,j)  + &
        Gamxxz(i,j,k) * p_gxyx(i,j) + Gamyxz(i,j,k) * p_gyyx(i,j) + Gamzxz(i,j,k) * p_gyzx(i,j)  + &
        Gamxxy(i,j,k) * p_gxzx(i,j) + Gamyxy(i,j,k) * p_gxzy(i,j) + Gamzxy(i,j,k) * p_gxzz(i,j)  + &
        Gamxxy(i,j,k) * p_gxxz(i,j) + Gamyxy(i,j,k) * p_gxyz(i,j) + Gamzxy(i,j,k) * p_gxzz(i,j)  + &
        Gamxyz(i,j,k) * p_gxxx(i,j) + Gamyyz(i,j,k) * p_gxyx(i,j) + Gamzyz(i,j,k) * p_gxzx(i,j)  + &
        Gamxxx(i,j,k) * p_gyzx(i,j) + Gamyxx(i,j,k) * p_gyzy(i,j) + Gamzxx(i,j,k) * p_gyzz(i,j) )+ &
        p_gupxz(i,j) *(                                                  &
        Gamxxx(i,j,k) * p_gxzz(i,j) + Gamyxx(i,j,k) * p_gyzz(i,j) + Gamzxx(i,j,k) * p_gzzz(i,j)  + &
        Gamxxz(i,j,k) * p_gxzx(i,j) + Gamyxz(i,j,k) * p_gyzx(i,j) + Gamzxz(i,j,k) * p_gzzx(i,j)  + &
        Gamxxz(i,j,k) * p_gxzx(i,j) + Gamyxz(i,j,k) * p_gxzy(i,j) + Gamzxz(i,j,k) * p_gxzz(i,j)  + &
        Gamxxz(i,j,k) * p_gxxz(i,j) + Gamyxz(i,j,k) * p_gxyz(i,j) + Gamzxz(i,j,k) * p_gxzz(i,j)  + &
        Gamxzz(i,j,k) * p_gxxx(i,j) + Gamyzz(i,j,k) * p_gxyx(i,j) + Gamzzz(i,j,k) * p_gxzx(i,j)  + &
        Gamxxx(i,j,k) * p_gzzx(i,j) + Gamyxx(i,j,k) * p_gzzy(i,j) + Gamzxx(i,j,k) * p_gzzz(i,j) )+ &
        p_gupyy(i,j) *(                                                  &
        Gamxxy(i,j,k) * p_gxyz(i,j) + Gamyxy(i,j,k) * p_gyyz(i,j) + Gamzxy(i,j,k) * p_gyzz(i,j)  + &
        Gamxyz(i,j,k) * p_gxyx(i,j) + Gamyyz(i,j,k) * p_gyyx(i,j) + Gamzyz(i,j,k) * p_gyzx(i,j)  + &
        Gamxxy(i,j,k) * p_gyzx(i,j) + Gamyxy(i,j,k) * p_gyzy(i,j) + Gamzxy(i,j,k) * p_gyzz(i,j) )+ &
        p_gupyz(i,j) *(                                                  &
        Gamxxy(i,j,k) * p_gxzz(i,j) + Gamyxy(i,j,k) * p_gyzz(i,j) + Gamzxy(i,j,k) * p_gzzz(i,j)  + &
        Gamxyz(i,j,k) * p_gxzx(i,j) + Gamyyz(i,j,k) * p_gyzx(i,j) + Gamzyz(i,j,k) * p_gzzx(i,j)  + &
        Gamxxz(i,j,k) * p_gyzx(i,j) + Gamyxz(i,j,k) * p_gyzy(i,j) + Gamzxz(i,j,k) * p_gyzz(i,j)  + &
        Gamxxz(i,j,k) * p_gxyz(i,j) + Gamyxz(i,j,k) * p_gyyz(i,j) + Gamzxz(i,j,k) * p_gyzz(i,j)  + &
        Gamxzz(i,j,k) * p_gxyx(i,j) + Gamyzz(i,j,k) * p_gyyx(i,j) + Gamzzz(i,j,k) * p_gyzx(i,j)  + &
        Gamxxy(i,j,k) * p_gzzx(i,j) + Gamyxy(i,j,k) * p_gzzy(i,j) + Gamzxy(i,j,k) * p_gzzz(i,j) )+ &
        p_gupzz(i,j) *(                                                  &
        Gamxxz(i,j,k) * p_gxzz(i,j) + Gamyxz(i,j,k) * p_gyzz(i,j) + Gamzxz(i,j,k) * p_gzzz(i,j)  + &
        Gamxzz(i,j,k) * p_gxzx(i,j) + Gamyzz(i,j,k) * p_gyzx(i,j) + Gamzzz(i,j,k) * p_gzzx(i,j)  + &
        Gamxxz(i,j,k) * p_gzzx(i,j) + Gamyxz(i,j,k) * p_gzzy(i,j) + Gamzxz(i,j,k) * p_gzzz(i,j) )

        Ryz(i,j,k) = HALF*(     - Ryz(i,j,k)                                   + &
        gxy(i,j,k) * p_Gamxz(i,j) + gyy(i,j,k) * p_Gamyz(i,j) + gyz(i,j,k) * p_Gamzz(i,j)     + &
        gxz(i,j,k) * p_Gamxy(i,j) + gyz(i,j,k) * p_Gamyy(i,j) + gzz(i,j,k) * p_Gamzy(i,j)     + &
        p_Gamxa(i,j) * p_gxzy(i,j) +  p_Gamya(i,j) * p_gyzy(i,j) +  p_Gamza(i,j) * p_gzzy(i,j)  + &
        p_Gamxa(i,j) * p_gxyz(i,j) +  p_Gamya(i,j) * p_gyyz(i,j) +  p_Gamza(i,j) * p_gyzz(i,j) )+ &
        p_gupxx(i,j) *(                                                  &
        Gamxxy(i,j,k) * p_gxxz(i,j) + Gamyxy(i,j,k) * p_gxyz(i,j) + Gamzxy(i,j,k) * p_gxzz(i,j)  + &
        Gamxxz(i,j,k) * p_gxxy(i,j) + Gamyxz(i,j,k) * p_gxyy(i,j) + Gamzxz(i,j,k) * p_gxzy(i,j)  + &
        Gamxxy(i,j,k) * p_gxzx(i,j) + Gamyxy(i,j,k) * p_gxzy(i,j) + Gamzxy(i,j,k) * p_gxzz(i,j) )+ &
        p_gupxy(i,j) *(                                                  &
        Gamxxy(i,j,k) * p_gxyz(i,j) + Gamyxy(i,j,k) * p_gyyz(i,j) + Gamzxy(i,j,k) * p_gyzz(i,j)  + &
        Gamxxz(i,j,k) * p_gxyy(i,j) + Gamyxz(i,j,k) * p_gyyy(i,j) + Gamzxz(i,j,k) * p_gyzy(i,j)  + &
        Gamxyy(i,j,k) * p_gxzx(i,j) + Gamyyy(i,j,k) * p_gxzy(i,j) + Gamzyy(i,j,k) * p_gxzz(i,j)  + &
        Gamxyy(i,j,k) * p_gxxz(i,j) + Gamyyy(i,j,k) * p_gxyz(i,j) + Gamzyy(i,j,k) * p_gxzz(i,j)  + &
        Gamxyz(i,j,k) * p_gxxy(i,j) + Gamyyz(i,j,k) * p_gxyy(i,j) + Gamzyz(i,j,k) * p_gxzy(i,j)  + &
        Gamxxy(i,j,k) * p_gyzx(i,j) + Gamyxy(i,j,k) * p_gyzy(i,j) + Gamzxy(i,j,k) * p_gyzz(i,j) )+ &
        p_gupxz(i,j) *(                                                  &
        Gamxxy(i,j,k) * p_gxzz(i,j) + Gamyxy(i,j,k) * p_gyzz(i,j) + Gamzxy(i,j,k) * p_gzzz(i,j)  + &
        Gamxxz(i,j,k) * p_gxzy(i,j) + Gamyxz(i,j,k) * p_gyzy(i,j) + Gamzxz(i,j,k) * p_gzzy(i,j)  + &
        Gamxyz(i,j,k) * p_gxzx(i,j) + Gamyyz(i,j,k) * p_gxzy(i,j) + Gamzyz(i,j,k) * p_gxzz(i,j)  + &
        Gamxyz(i,j,k) * p_gxxz(i,j) + Gamyyz(i,j,k) * p_gxyz(i,j) + Gamzyz(i,j,k) * p_gxzz(i,j)  + &
        Gamxzz(i,j,k) * p_gxxy(i,j) + Gamyzz(i,j,k) * p_gxyy(i,j) + Gamzzz(i,j,k) * p_gxzy(i,j)  + &
        Gamxxy(i,j,k) * p_gzzx(i,j) + Gamyxy(i,j,k) * p_gzzy(i,j) + Gamzxy(i,j,k) * p_gzzz(i,j) )+ &
        p_gupyy(i,j) *(                                                  &
        Gamxyy(i,j,k) * p_gxyz(i,j) + Gamyyy(i,j,k) * p_gyyz(i,j) + Gamzyy(i,j,k) * p_gyzz(i,j)  + &
        Gamxyz(i,j,k) * p_gxyy(i,j) + Gamyyz(i,j,k) * p_gyyy(i,j) + Gamzyz(i,j,k) * p_gyzy(i,j)  + &
        Gamxyy(i,j,k) * p_gyzx(i,j) + Gamyyy(i,j,k) * p_gyzy(i,j) + Gamzyy(i,j,k) * p_gyzz(i,j) )+ &
        p_gupyz(i,j) *(                                                  &
        Gamxyy(i,j,k) * p_gxzz(i,j) + Gamyyy(i,j,k) * p_gyzz(i,j) + Gamzyy(i,j,k) * p_gzzz(i,j)  + &
        Gamxyz(i,j,k) * p_gxzy(i,j) + Gamyyz(i,j,k) * p_gyzy(i,j) + Gamzyz(i,j,k) * p_gzzy(i,j)  + &
        Gamxyz(i,j,k) * p_gyzx(i,j) + Gamyyz(i,j,k) * p_gyzy(i,j) + Gamzyz(i,j,k) * p_gyzz(i,j)  + &
        Gamxyz(i,j,k) * p_gxyz(i,j) + Gamyyz(i,j,k) * p_gyyz(i,j) + Gamzyz(i,j,k) * p_gyzz(i,j)  + &
        Gamxzz(i,j,k) * p_gxyy(i,j) + Gamyzz(i,j,k) * p_gyyy(i,j) + Gamzzz(i,j,k) * p_gyzy(i,j)  + &
        Gamxyy(i,j,k) * p_gzzx(i,j) + Gamyyy(i,j,k) * p_gzzy(i,j) + Gamzyy(i,j,k) * p_gzzz(i,j) )+ &
        p_gupzz(i,j) *(                                                  &
        Gamxyz(i,j,k) * p_gxzz(i,j) + Gamyyz(i,j,k) * p_gyzz(i,j) + Gamzyz(i,j,k) * p_gzzz(i,j)  + &
        Gamxzz(i,j,k) * p_gxzy(i,j) + Gamyzz(i,j,k) * p_gyzy(i,j) + Gamzzz(i,j,k) * p_gzzy(i,j)  + &
        Gamxyz(i,j,k) * p_gzzx(i,j) + Gamyyz(i,j,k) * p_gzzy(i,j) + Gamzyz(i,j,k) * p_gzzz(i,j) )
      end do
    end do
  call fdderivs_plane(chi, p_fxx, p_fxy, p_fxz, p_fyy, p_fyz, p_fzz, SYM, SYM, SYM)
  do j = 1, ex(2)
  do i = 1, ex(1)
  chin1 = chi(i,j,k) + ONE
  s_gxx = dxx(i,j,k) + ONE
  s_gyy = dyy(i,j,k) + ONE
  s_gzz = dzz(i,j,k) + ONE
  s_fxx = p_fxx(i,j) - Gamxxx(i,j,k) * p_chix(i,j) - Gamyxx(i,j,k) * p_chiy(i,j) - Gamzxx(i,j,k) * p_chiz(i,j)
  s_fxy = p_fxy(i,j) - Gamxxy(i,j,k) * p_chix(i,j) - Gamyxy(i,j,k) * p_chiy(i,j) - Gamzxy(i,j,k) * p_chiz(i,j)
  s_fxz = p_fxz(i,j) - Gamxxz(i,j,k) * p_chix(i,j) - Gamyxz(i,j,k) * p_chiy(i,j) - Gamzxz(i,j,k) * p_chiz(i,j)
  s_fyy = p_fyy(i,j) - Gamxyy(i,j,k) * p_chix(i,j) - Gamyyy(i,j,k) * p_chiy(i,j) - Gamzyy(i,j,k) * p_chiz(i,j)
  s_fyz = p_fyz(i,j) - Gamxyz(i,j,k) * p_chix(i,j) - Gamyyz(i,j,k) * p_chiy(i,j) - Gamzyz(i,j,k) * p_chiz(i,j)
  s_fzz = p_fzz(i,j) - Gamxzz(i,j,k) * p_chix(i,j) - Gamyzz(i,j,k) * p_chiy(i,j) - Gamzzz(i,j,k) * p_chiz(i,j)
  ! Store D^l D_l chi - 3/(2*chi) D^l chi D_l chi in f
  f =        p_gupxx(i,j) * ( s_fxx - F3o2/chin1 * p_chix(i,j) * p_chix(i,j) ) + &
  p_gupyy(i,j) * ( s_fyy - F3o2/chin1 * p_chiy(i,j) * p_chiy(i,j) ) + &
  p_gupzz(i,j) * ( s_fzz - F3o2/chin1 * p_chiz(i,j) * p_chiz(i,j) ) + &
  TWO * p_gupxy(i,j) * ( s_fxy - F3o2/chin1 * p_chix(i,j) * p_chiy(i,j) ) + &
  TWO * p_gupxz(i,j) * ( s_fxz - F3o2/chin1 * p_chix(i,j) * p_chiz(i,j) ) + &
  TWO * p_gupyz(i,j) * ( s_fyz - F3o2/chin1 * p_chiy(i,j) * p_chiz(i,j) )
  ! Add chi part to Ricci tensor:
  Rxx(i,j,k) = Rxx(i,j,k) + (s_fxx - p_chix(i,j)*p_chix(i,j)/chin1/TWO + s_gxx * f)/chin1/TWO
  Ryy(i,j,k) = Ryy(i,j,k) + (s_fyy - p_chiy(i,j)*p_chiy(i,j)/chin1/TWO + s_gyy * f)/chin1/TWO
  Rzz(i,j,k) = Rzz(i,j,k) + (s_fzz - p_chiz(i,j)*p_chiz(i,j)/chin1/TWO + s_gzz * f)/chin1/TWO
  Rxy(i,j,k) = Rxy(i,j,k) + (s_fxy - p_chix(i,j)*p_chiy(i,j)/chin1/TWO + gxy(i,j,k) * f)/chin1/TWO
  Rxz(i,j,k) = Rxz(i,j,k) + (s_fxz - p_chix(i,j)*p_chiz(i,j)/chin1/TWO + gxz(i,j,k) * f)/chin1/TWO
  Ryz(i,j,k) = Ryz(i,j,k) + (s_fyz - p_chiy(i,j)*p_chiz(i,j)/chin1/TWO + gyz(i,j,k) * f)/chin1/TWO
  end do
  end do
  call fdderivs_plane(Lap, p_fxx, p_fxy, p_fxz, p_fyy, p_fyz, p_fzz, SYM, SYM, SYM)
  do j = 1, ex(2)
  do i = 1, ex(1)
  chin1 = chi(i,j,k) + ONE
  s_gxx = dxx(i,j,k) + ONE
  s_gyy = dyy(i,j,k) + ONE
  s_gzz = dzz(i,j,k) + ONE
  s_gxxx = (p_gupxx(i,j) * p_chix(i,j) + p_gupxy(i,j) * p_chiy(i,j) + p_gupxz(i,j) * p_chiz(i,j))/chin1
  s_gxxy = (p_gupxy(i,j) * p_chix(i,j) + p_gupyy(i,j) * p_chiy(i,j) + p_gupyz(i,j) * p_chiz(i,j))/chin1
  s_gxxz = (p_gupxz(i,j) * p_chix(i,j) + p_gupyz(i,j) * p_chiy(i,j) + p_gupzz(i,j) * p_chiz(i,j))/chin1
  ! now get physical second kind of connection
  Gamxxx(i,j,k) = Gamxxx(i,j,k) - ( (p_chix(i,j) + p_chix(i,j))/chin1 - s_gxx * s_gxxx )*HALF
  Gamyxx(i,j,k) = Gamyxx(i,j,k) - (                     - s_gxx * s_gxxy )*HALF
  Gamzxx(i,j,k) = Gamzxx(i,j,k) - (                     - s_gxx * s_gxxz )*HALF
  Gamxyy(i,j,k) = Gamxyy(i,j,k) - (                     - s_gyy * s_gxxx )*HALF
  Gamyyy(i,j,k) = Gamyyy(i,j,k) - ( (p_chiy(i,j) + p_chiy(i,j))/chin1 - s_gyy * s_gxxy )*HALF
  Gamzyy(i,j,k) = Gamzyy(i,j,k) - (                     - s_gyy * s_gxxz )*HALF
  Gamxzz(i,j,k) = Gamxzz(i,j,k) - (                     - s_gzz * s_gxxx )*HALF
  Gamyzz(i,j,k) = Gamyzz(i,j,k) - (                     - s_gzz * s_gxxy )*HALF
  Gamzzz(i,j,k) = Gamzzz(i,j,k) - ( (p_chiz(i,j) + p_chiz(i,j))/chin1 - s_gzz * s_gxxz )*HALF
  Gamxxy(i,j,k) = Gamxxy(i,j,k) - (  p_chiy(i,j)        /chin1 - gxy(i,j,k) * s_gxxx )*HALF
  Gamyxy(i,j,k) = Gamyxy(i,j,k) - (         p_chix(i,j) /chin1 - gxy(i,j,k) * s_gxxy )*HALF
  Gamzxy(i,j,k) = Gamzxy(i,j,k) - (                     - gxy(i,j,k) * s_gxxz )*HALF
  Gamxxz(i,j,k) = Gamxxz(i,j,k) - (  p_chiz(i,j)        /chin1 - gxz(i,j,k) * s_gxxx )*HALF
  Gamyxz(i,j,k) = Gamyxz(i,j,k) - (                     - gxz(i,j,k) * s_gxxy )*HALF
  Gamzxz(i,j,k) = Gamzxz(i,j,k) - (         p_chix(i,j) /chin1 - gxz(i,j,k) * s_gxxz )*HALF
  Gamxyz(i,j,k) = Gamxyz(i,j,k) - (                     - gyz(i,j,k) * s_gxxx )*HALF
  Gamyyz(i,j,k) = Gamyyz(i,j,k) - (  p_chiz(i,j)        /chin1 - gyz(i,j,k) * s_gxxy )*HALF
  Gamzyz(i,j,k) = Gamzyz(i,j,k) - (         p_chiy(i,j) /chin1 - gyz(i,j,k) * s_gxxz )*HALF

  p_fxx(i,j) = p_fxx(i,j) - Gamxxx(i,j,k)*p_Lapx(i,j) - Gamyxx(i,j,k)*p_Lapy(i,j) - Gamzxx(i,j,k)*p_Lapz(i,j)
  p_fyy(i,j) = p_fyy(i,j) - Gamxyy(i,j,k)*p_Lapx(i,j) - Gamyyy(i,j,k)*p_Lapy(i,j) - Gamzyy(i,j,k)*p_Lapz(i,j)
  p_fzz(i,j) = p_fzz(i,j) - Gamxzz(i,j,k)*p_Lapx(i,j) - Gamyzz(i,j,k)*p_Lapy(i,j) - Gamzzz(i,j,k)*p_Lapz(i,j)
  p_fxy(i,j) = p_fxy(i,j) - Gamxxy(i,j,k)*p_Lapx(i,j) - Gamyxy(i,j,k)*p_Lapy(i,j) - Gamzxy(i,j,k)*p_Lapz(i,j)
  p_fxz(i,j) = p_fxz(i,j) - Gamxxz(i,j,k)*p_Lapx(i,j) - Gamyxz(i,j,k)*p_Lapy(i,j) - Gamzxz(i,j,k)*p_Lapz(i,j)
  p_fyz(i,j) = p_fyz(i,j) - Gamxyz(i,j,k)*p_Lapx(i,j) - Gamyyz(i,j,k)*p_Lapy(i,j) - Gamzyz(i,j,k)*p_Lapz(i,j)
  end do
  end do
  do j = 1, ex(2)
  do i = 1, ex(1)
  alpn1   = Lap(i,j,k) + ONE
  chin1   = chi(i,j,k) + ONE
  s_gxx   = dxx(i,j,k) + ONE
  s_gyy   = dyy(i,j,k) + ONE
  s_gzz   = dzz(i,j,k) + ONE
  div_beta = p_betaxx(i,j) + p_betayy(i,j) + p_betazz(i,j)
  ! store D^i D_i Lap in trK_rhs upto chi
  trK_rhs(i,j,k) =    p_gupxx(i,j) * p_fxx(i,j) + p_gupyy(i,j) * p_fyy(i,j) + p_gupzz(i,j) * p_fzz(i,j) + &
  TWO* ( p_gupxy(i,j) * p_fxy(i,j) + p_gupxz(i,j) * p_fxz(i,j) + p_gupyz(i,j) * p_fyz(i,j) )
  !! follow bam code
  S =  chin1 * ( p_gupxx(i,j) * Sxx(i,j,k) + p_gupyy(i,j) * Syy(i,j,k) + p_gupzz(i,j) * Szz(i,j,k) + &
  TWO * ( p_gupxy(i,j) * Sxy(i,j,k) + p_gupxz(i,j) * Sxz(i,j,k) + p_gupyz(i,j) * Syz(i,j,k) ) )
  f = F2o3 * trK(i,j,k) * trK(i,j,k) -(&
  p_gupxx(i,j) * ( &
  p_gupxx(i,j) * Axx(i,j,k) * Axx(i,j,k) + p_gupyy(i,j) * Axy(i,j,k) * Axy(i,j,k) + p_gupzz(i,j) * Axz(i,j,k) * Axz(i,j,k) + &
  TWO * (p_gupxy(i,j) * Axx(i,j,k) * Axy(i,j,k) + p_gupxz(i,j) * Axx(i,j,k) * Axz(i,j,k) + p_gupyz(i,j) * Axy(i,j,k) * Axz(i,j,k) ) ) + &
  p_gupyy(i,j) * ( &
  p_gupxx(i,j) * Axy(i,j,k) * Axy(i,j,k) + p_gupyy(i,j) * Ayy(i,j,k) * Ayy(i,j,k) + p_gupzz(i,j) * Ayz(i,j,k) * Ayz(i,j,k) + &
  TWO * (p_gupxy(i,j) * Axy(i,j,k) * Ayy(i,j,k) + p_gupxz(i,j) * Axy(i,j,k) * Ayz(i,j,k) + p_gupyz(i,j) * Ayy(i,j,k) * Ayz(i,j,k) ) ) + &
  p_gupzz(i,j) * ( &
  p_gupxx(i,j) * Axz(i,j,k) * Axz(i,j,k) + p_gupyy(i,j) * Ayz(i,j,k) * Ayz(i,j,k) + p_gupzz(i,j) * Azz(i,j,k) * Azz(i,j,k) + &
  TWO * (p_gupxy(i,j) * Axz(i,j,k) * Ayz(i,j,k) + p_gupxz(i,j) * Axz(i,j,k) * Azz(i,j,k) + p_gupyz(i,j) * Ayz(i,j,k) * Azz(i,j,k) ) ) + &
  TWO * ( &
  p_gupxy(i,j) * ( &
  p_gupxx(i,j) * Axx(i,j,k) * Axy(i,j,k) + p_gupyy(i,j) * Axy(i,j,k) * Ayy(i,j,k) + p_gupzz(i,j) * Axz(i,j,k) * Ayz(i,j,k) + &
  p_gupxy(i,j) * (Axx(i,j,k) * Ayy(i,j,k) + Axy(i,j,k) * Axy(i,j,k)) + &
  p_gupxz(i,j) * (Axx(i,j,k) * Ayz(i,j,k) + Axz(i,j,k) * Axy(i,j,k)) + &
  p_gupyz(i,j) * (Axy(i,j,k) * Ayz(i,j,k) + Axz(i,j,k) * Ayy(i,j,k)) ) + &
  p_gupxz(i,j) * ( &
  p_gupxx(i,j) * Axx(i,j,k) * Axz(i,j,k) + p_gupyy(i,j) * Axy(i,j,k) * Ayz(i,j,k) + p_gupzz(i,j) * Axz(i,j,k) * Azz(i,j,k) + &
  p_gupxy(i,j) * (Axx(i,j,k) * Ayz(i,j,k) + Axy(i,j,k) * Axz(i,j,k)) + &
  p_gupxz(i,j) * (Axx(i,j,k) * Azz(i,j,k) + Axz(i,j,k) * Axz(i,j,k)) + &
  p_gupyz(i,j) * (Axy(i,j,k) * Azz(i,j,k) + Axz(i,j,k) * Ayz(i,j,k)) ) + &
  p_gupyz(i,j) * ( &
  p_gupxx(i,j) * Axy(i,j,k) * Axz(i,j,k) + p_gupyy(i,j) * Ayy(i,j,k) * Ayz(i,j,k) + p_gupzz(i,j) * Ayz(i,j,k) * Azz(i,j,k) + &
  p_gupxy(i,j) * (Axy(i,j,k) * Ayz(i,j,k) + Ayy(i,j,k) * Axz(i,j,k)) + &
  p_gupxz(i,j) * (Axy(i,j,k) * Azz(i,j,k) + Ayz(i,j,k) * Axz(i,j,k)) + &
  p_gupyz(i,j) * (Ayy(i,j,k) * Azz(i,j,k) + Ayz(i,j,k) * Ayz(i,j,k)) ) )) -1.6d1*PI*rho(i,j,k) + EIGHT * PI * S
  f = - F1o3 *(  p_gupxx(i,j) * p_fxx(i,j) + p_gupyy(i,j) * p_fyy(i,j) + p_gupzz(i,j) * p_fzz(i,j) + &
  TWO* ( p_gupxy(i,j) * p_fxy(i,j) + p_gupxz(i,j) * p_fxz(i,j) + p_gupyz(i,j) * p_fyz(i,j) ) + alpn1/chin1*f)

  s_fxx = alpn1 * (Rxx(i,j,k) - EIGHT * PI * Sxx(i,j,k)) - p_fxx(i,j)
  s_fxy = alpn1 * (Rxy(i,j,k) - EIGHT * PI * Sxy(i,j,k)) - p_fxy(i,j)
  s_fxz = alpn1 * (Rxz(i,j,k) - EIGHT * PI * Sxz(i,j,k)) - p_fxz(i,j)
  s_fyy = alpn1 * (Ryy(i,j,k) - EIGHT * PI * Syy(i,j,k)) - p_fyy(i,j)
  s_fyz = alpn1 * (Ryz(i,j,k) - EIGHT * PI * Syz(i,j,k)) - p_fyz(i,j)
  s_fzz = alpn1 * (Rzz(i,j,k) - EIGHT * PI * Szz(i,j,k)) - p_fzz(i,j)

  Axx_rhs(i,j,k) = s_fxx - s_gxx * f
  Ayy_rhs(i,j,k) = s_fyy - s_gyy * f
  Azz_rhs(i,j,k) = s_fzz - s_gzz * f
  Axy_rhs(i,j,k) = s_fxy - gxy(i,j,k) * f
  Axz_rhs(i,j,k) = s_fxz - gxz(i,j,k) * f
  Ayz_rhs(i,j,k) = s_fyz - gyz(i,j,k) * f

  ! Now: store A_il A^l_j into fij:
  s_fxx =       p_gupxx(i,j) * Axx(i,j,k) * Axx(i,j,k) + p_gupyy(i,j) * Axy(i,j,k) * Axy(i,j,k) + p_gupzz(i,j) * Axz(i,j,k) * Axz(i,j,k) + &
  TWO * (p_gupxy(i,j) * Axx(i,j,k) * Axy(i,j,k) + p_gupxz(i,j) * Axx(i,j,k) * Axz(i,j,k) + p_gupyz(i,j) * Axy(i,j,k) * Axz(i,j,k))
  s_fyy =       p_gupxx(i,j) * Axy(i,j,k) * Axy(i,j,k) + p_gupyy(i,j) * Ayy(i,j,k) * Ayy(i,j,k) + p_gupzz(i,j) * Ayz(i,j,k) * Ayz(i,j,k) + &
  TWO * (p_gupxy(i,j) * Axy(i,j,k) * Ayy(i,j,k) + p_gupxz(i,j) * Axy(i,j,k) * Ayz(i,j,k) + p_gupyz(i,j) * Ayy(i,j,k) * Ayz(i,j,k))
  s_fzz =       p_gupxx(i,j) * Axz(i,j,k) * Axz(i,j,k) + p_gupyy(i,j) * Ayz(i,j,k) * Ayz(i,j,k) + p_gupzz(i,j) * Azz(i,j,k) * Azz(i,j,k) + &
  TWO * (p_gupxy(i,j) * Axz(i,j,k) * Ayz(i,j,k) + p_gupxz(i,j) * Axz(i,j,k) * Azz(i,j,k) + p_gupyz(i,j) * Ayz(i,j,k) * Azz(i,j,k))
  s_fxy =       p_gupxx(i,j) * Axx(i,j,k) * Axy(i,j,k) + p_gupyy(i,j) * Axy(i,j,k) * Ayy(i,j,k) + p_gupzz(i,j) * Axz(i,j,k) * Ayz(i,j,k) + &
  p_gupxy(i,j) *(Axx(i,j,k) * Ayy(i,j,k) + Axy(i,j,k) * Axy(i,j,k))                            + &
  p_gupxz(i,j) *(Axx(i,j,k) * Ayz(i,j,k) + Axz(i,j,k) * Axy(i,j,k))                            + &
  p_gupyz(i,j) *(Axy(i,j,k) * Ayz(i,j,k) + Axz(i,j,k) * Ayy(i,j,k))
  s_fxz =       p_gupxx(i,j) * Axx(i,j,k) * Axz(i,j,k) + p_gupyy(i,j) * Axy(i,j,k) * Ayz(i,j,k) + p_gupzz(i,j) * Axz(i,j,k) * Azz(i,j,k) + &
  p_gupxy(i,j) *(Axx(i,j,k) * Ayz(i,j,k) + Axy(i,j,k) * Axz(i,j,k))                            + &
  p_gupxz(i,j) *(Axx(i,j,k) * Azz(i,j,k) + Axz(i,j,k) * Axz(i,j,k))                            + &
  p_gupyz(i,j) *(Axy(i,j,k) * Azz(i,j,k) + Axz(i,j,k) * Ayz(i,j,k))
  s_fyz =       p_gupxx(i,j) * Axy(i,j,k) * Axz(i,j,k) + p_gupyy(i,j) * Ayy(i,j,k) * Ayz(i,j,k) + p_gupzz(i,j) * Ayz(i,j,k) * Azz(i,j,k) + &
  p_gupxy(i,j) *(Axy(i,j,k) * Ayz(i,j,k) + Ayy(i,j,k) * Axz(i,j,k))                            + &
  p_gupxz(i,j) *(Axy(i,j,k) * Azz(i,j,k) + Ayz(i,j,k) * Axz(i,j,k))                            + &
  p_gupyz(i,j) *(Ayy(i,j,k) * Azz(i,j,k) + Ayz(i,j,k) * Ayz(i,j,k))

  f = chin1
  ! store D^i D_i Lap in trK_rhs
  trK_rhs(i,j,k) = f*trK_rhs(i,j,k)

  Axx_rhs(i,j,k) =           f * Axx_rhs(i,j,k) + alpn1 * (trK(i,j,k) * Axx(i,j,k) - TWO * s_fxx)  + &
  TWO * (  Axx(i,j,k) * p_betaxx(i,j) +   Axy(i,j,k) * p_betayx(i,j) +   Axz(i,j,k) * p_betazx(i,j) )- &
  F2o3 * Axx(i,j,k) * div_beta
  Ayy_rhs(i,j,k) =           f * Ayy_rhs(i,j,k) + alpn1 * (trK(i,j,k) * Ayy(i,j,k) - TWO * s_fyy)  + &
  TWO * (  Axy(i,j,k) * p_betaxy(i,j) +   Ayy(i,j,k) * p_betayy(i,j) +   Ayz(i,j,k) * p_betazy(i,j) )- &
  F2o3 * Ayy(i,j,k) * div_beta
  Azz_rhs(i,j,k) =           f * Azz_rhs(i,j,k) + alpn1 * (trK(i,j,k) * Azz(i,j,k) - TWO * s_fzz)  + &
  TWO * (  Axz(i,j,k) * p_betaxz(i,j) +   Ayz(i,j,k) * p_betayz(i,j) +   Azz(i,j,k) * p_betazz(i,j) )- &
  F2o3 * Azz(i,j,k) * div_beta
  Axy_rhs(i,j,k) =           f * Axy_rhs(i,j,k) + alpn1 *( trK(i,j,k) * Axy(i,j,k)  - TWO * s_fxy )+ &
  Axx(i,j,k) * p_betaxy(i,j)                  +   Axz(i,j,k) * p_betazy(i,j)  + &
  Ayy(i,j,k) * p_betayx(i,j) +   Ayz(i,j,k) * p_betazx(i,j)  + &
  F1o3 * Axy(i,j,k) * div_beta                -   Axy(i,j,k) * p_betazz(i,j)
  Ayz_rhs(i,j,k) =           f * Ayz_rhs(i,j,k) + alpn1 *( trK(i,j,k) * Ayz(i,j,k)  - TWO * s_fyz )+ &
  Axy(i,j,k) * p_betaxz(i,j) +   Ayy(i,j,k) * p_betayz(i,j)                   + &
  Axz(i,j,k) * p_betaxy(i,j)                  +   Azz(i,j,k) * p_betazy(i,j)  + &
  F1o3 * Ayz(i,j,k) * div_beta                -   Ayz(i,j,k) * p_betaxx(i,j)

  Axz_rhs(i,j,k) =           f * Axz_rhs(i,j,k) + alpn1 *( trK(i,j,k) * Axz(i,j,k)  - TWO * s_fxz )+ &
  Axx(i,j,k) * p_betaxz(i,j) +   Axy(i,j,k) * p_betayz(i,j)                   + &
  Ayz(i,j,k) * p_betayx(i,j) +   Azz(i,j,k) * p_betazx(i,j)  + &
  F1o3 * Axz(i,j,k) * div_beta                -   Axz(i,j,k) * p_betayy(i,j)      !rhs for Aij

  ! Compute trace of S_ij
  S =  f * ( p_gupxx(i,j) * Sxx(i,j,k) + p_gupyy(i,j) * Syy(i,j,k) + p_gupzz(i,j) * Szz(i,j,k) + &
  TWO * ( p_gupxy(i,j) * Sxy(i,j,k) + p_gupxz(i,j) * Sxz(i,j,k) + p_gupyz(i,j) * Syz(i,j,k) ) )

  trK_rhs(i,j,k) = - trK_rhs(i,j,k) + alpn1 *( F1o3 * trK(i,j,k) * trK(i,j,k)         + &
  p_gupxx(i,j) * s_fxx + p_gupyy(i,j) * s_fyy + p_gupzz(i,j) * s_fzz   + &
  TWO * ( p_gupxy(i,j) * s_fxy + p_gupxz(i,j) * s_fxz + p_gupyz(i,j) * s_fyz ) + &
  FOUR * PI * ( rho(i,j,k) + S ))                                !rhs for trK

  !!!! gauge variable part
  Lap_rhs(i,j,k) = -TWO*alpn1*trK(i,j,k)
  betax_rhs(i,j,k) = FF*dtSfx(i,j,k)
  betay_rhs(i,j,k) = FF*dtSfy(i,j,k)
  betaz_rhs(i,j,k) = FF*dtSfz(i,j,k)

  dtSfx_rhs(i,j,k) = Gamx_rhs(i,j,k) - eta*dtSfx(i,j,k)
  dtSfy_rhs(i,j,k) = Gamy_rhs(i,j,k) - eta*dtSfy(i,j,k)
  dtSfz_rhs(i,j,k) = Gamz_rhs(i,j,k) - eta*dtSfz(i,j,k)
  end do
  end do
    if (co == 0) then
    do j = 1, ex(2)
    do i = 1, ex(1)
    chin1 = chi(i,j,k) + ONE
    hm =   p_gupxx(i,j) * Rxx(i,j,k) + p_gupyy(i,j) * Ryy(i,j,k) + p_gupzz(i,j) * Rzz(i,j,k) + &
    TWO* ( p_gupxy(i,j) * Rxy(i,j,k) + p_gupxz(i,j) * Rxz(i,j,k) + p_gupyz(i,j) * Ryz(i,j,k) )
    hm = chin1*hm + F2o3 * trK(i,j,k) * trK(i,j,k) -(&
    p_gupxx(i,j) * ( &
    p_gupxx(i,j) * Axx(i,j,k) * Axx(i,j,k) + p_gupyy(i,j) * Axy(i,j,k) * Axy(i,j,k) + p_gupzz(i,j) * Axz(i,j,k) * Axz(i,j,k) + &
    TWO * (p_gupxy(i,j) * Axx(i,j,k) * Axy(i,j,k) + p_gupxz(i,j) * Axx(i,j,k) * Axz(i,j,k) + p_gupyz(i,j) * Axy(i,j,k) * Axz(i,j,k) ) ) + &
    p_gupyy(i,j) * ( &
    p_gupxx(i,j) * Axy(i,j,k) * Axy(i,j,k) + p_gupyy(i,j) * Ayy(i,j,k) * Ayy(i,j,k) + p_gupzz(i,j) * Ayz(i,j,k) * Ayz(i,j,k) + &
    TWO * (p_gupxy(i,j) * Axy(i,j,k) * Ayy(i,j,k) + p_gupxz(i,j) * Axy(i,j,k) * Ayz(i,j,k) + p_gupyz(i,j) * Ayy(i,j,k) * Ayz(i,j,k) ) ) + &
    p_gupzz(i,j) * ( &
    p_gupxx(i,j) * Axz(i,j,k) * Axz(i,j,k) + p_gupyy(i,j) * Ayz(i,j,k) * Ayz(i,j,k) + p_gupzz(i,j) * Azz(i,j,k) * Azz(i,j,k) + &
    TWO * (p_gupxy(i,j) * Axz(i,j,k) * Ayz(i,j,k) + p_gupxz(i,j) * Axz(i,j,k) * Azz(i,j,k) + p_gupyz(i,j) * Ayz(i,j,k) * Azz(i,j,k) ) ) + &
    TWO * ( &
    p_gupxy(i,j) * ( &
    p_gupxx(i,j) * Axx(i,j,k) * Axy(i,j,k) + p_gupyy(i,j) * Axy(i,j,k) * Ayy(i,j,k) + p_gupzz(i,j) * Axz(i,j,k) * Ayz(i,j,k) + &
    p_gupxy(i,j) * (Axx(i,j,k) * Ayy(i,j,k) + Axy(i,j,k) * Axy(i,j,k)) + &
    p_gupxz(i,j) * (Axx(i,j,k) * Ayz(i,j,k) + Axz(i,j,k) * Axy(i,j,k)) + &
    p_gupyz(i,j) * (Axy(i,j,k) * Ayz(i,j,k) + Axz(i,j,k) * Ayy(i,j,k)) ) + &
    p_gupxz(i,j) * ( &
    p_gupxx(i,j) * Axx(i,j,k) * Axz(i,j,k) + p_gupyy(i,j) * Axy(i,j,k) * Ayz(i,j,k) + p_gupzz(i,j) * Axz(i,j,k) * Azz(i,j,k) + &
    p_gupxy(i,j) * (Axx(i,j,k) * Ayz(i,j,k) + Axy(i,j,k) * Axz(i,j,k)) + &
    p_gupxz(i,j) * (Axx(i,j,k) * Azz(i,j,k) + Axz(i,j,k) * Axz(i,j,k)) + &
    p_gupyz(i,j) * (Axy(i,j,k) * Azz(i,j,k) + Axz(i,j,k) * Ayz(i,j,k)) ) + &
    p_gupyz(i,j) * ( &
    p_gupxx(i,j) * Axy(i,j,k) * Axz(i,j,k) + p_gupyy(i,j) * Ayy(i,j,k) * Ayz(i,j,k) + p_gupzz(i,j) * Ayz(i,j,k) * Azz(i,j,k) + &
    p_gupxy(i,j) * (Axy(i,j,k) * Ayz(i,j,k) + Ayy(i,j,k) * Axz(i,j,k)) + &
    p_gupxz(i,j) * (Axy(i,j,k) * Azz(i,j,k) + Ayz(i,j,k) * Axz(i,j,k)) + &
    p_gupyz(i,j) * (Ayy(i,j,k) * Azz(i,j,k) + Ayz(i,j,k) * Ayz(i,j,k)) ) ))- F16 * PI * rho(i,j,k)
    ham_Res(i,j,k) = hm
    end do
    end do
  call fderivs_plane(Axx, p_gxxx, p_gxxy, p_gxxz, SYM, SYM, SYM)
  call fderivs_plane(Axy, p_gxyx, p_gxyy, p_gxyz, ANTI, ANTI, SYM)
  call fderivs_plane(Axz, p_gxzx, p_gxzy, p_gxzz, ANTI, SYM, ANTI)
  call fderivs_plane(Ayy, p_gyyx, p_gyyy, p_gyyz, SYM, SYM, SYM)
  call fderivs_plane(Ayz, p_gyzx, p_gyzy, p_gyzz, SYM, ANTI, ANTI)
  call fderivs_plane(Azz, p_gzzx, p_gzzy, p_gzzz, SYM, SYM, SYM)
    do j = 1, ex(2)
    do i = 1, ex(1)
    chin1 = chi(i,j,k) + ONE
    p_gxxx(i,j) = p_gxxx(i,j) - (  Gamxxx(i,j,k) * Axx(i,j,k) + Gamyxx(i,j,k) * Axy(i,j,k) + Gamzxx(i,j,k) * Axz(i,j,k) &
    + Gamxxx(i,j,k) * Axx(i,j,k) + Gamyxx(i,j,k) * Axy(i,j,k) + Gamzxx(i,j,k) * Axz(i,j,k)) - p_chix(i,j)*Axx(i,j,k)/chin1
    p_gxyx(i,j) = p_gxyx(i,j) - (  Gamxxy(i,j,k) * Axx(i,j,k) + Gamyxy(i,j,k) * Axy(i,j,k) + Gamzxy(i,j,k) * Axz(i,j,k) &
    + Gamxxx(i,j,k) * Axy(i,j,k) + Gamyxx(i,j,k) * Ayy(i,j,k) + Gamzxx(i,j,k) * Ayz(i,j,k)) - p_chix(i,j)*Axy(i,j,k)/chin1
    p_gxzx(i,j) = p_gxzx(i,j) - (  Gamxxz(i,j,k) * Axx(i,j,k) + Gamyxz(i,j,k) * Axy(i,j,k) + Gamzxz(i,j,k) * Axz(i,j,k) &
    + Gamxxx(i,j,k) * Axz(i,j,k) + Gamyxx(i,j,k) * Ayz(i,j,k) + Gamzxx(i,j,k) * Azz(i,j,k)) - p_chix(i,j)*Axz(i,j,k)/chin1
    p_gyyx(i,j) = p_gyyx(i,j) - (  Gamxxy(i,j,k) * Axy(i,j,k) + Gamyxy(i,j,k) * Ayy(i,j,k) + Gamzxy(i,j,k) * Ayz(i,j,k) &
    + Gamxxy(i,j,k) * Axy(i,j,k) + Gamyxy(i,j,k) * Ayy(i,j,k) + Gamzxy(i,j,k) * Ayz(i,j,k)) - p_chix(i,j)*Ayy(i,j,k)/chin1
    p_gyzx(i,j) = p_gyzx(i,j) - (  Gamxxz(i,j,k) * Axy(i,j,k) + Gamyxz(i,j,k) * Ayy(i,j,k) + Gamzxz(i,j,k) * Ayz(i,j,k) &
    + Gamxxy(i,j,k) * Axz(i,j,k) + Gamyxy(i,j,k) * Ayz(i,j,k) + Gamzxy(i,j,k) * Azz(i,j,k)) - p_chix(i,j)*Ayz(i,j,k)/chin1
    p_gzzx(i,j) = p_gzzx(i,j) - (  Gamxxz(i,j,k) * Axz(i,j,k) + Gamyxz(i,j,k) * Ayz(i,j,k) + Gamzxz(i,j,k) * Azz(i,j,k) &
    + Gamxxz(i,j,k) * Axz(i,j,k) + Gamyxz(i,j,k) * Ayz(i,j,k) + Gamzxz(i,j,k) * Azz(i,j,k)) - p_chix(i,j)*Azz(i,j,k)/chin1
    p_gxxy(i,j) = p_gxxy(i,j) - (  Gamxxy(i,j,k) * Axx(i,j,k) + Gamyxy(i,j,k) * Axy(i,j,k) + Gamzxy(i,j,k) * Axz(i,j,k) &
    + Gamxxy(i,j,k) * Axx(i,j,k) + Gamyxy(i,j,k) * Axy(i,j,k) + Gamzxy(i,j,k) * Axz(i,j,k)) - p_chiy(i,j)*Axx(i,j,k)/chin1
    p_gxyy(i,j) = p_gxyy(i,j) - (  Gamxyy(i,j,k) * Axx(i,j,k) + Gamyyy(i,j,k) * Axy(i,j,k) + Gamzyy(i,j,k) * Axz(i,j,k) &
    + Gamxxy(i,j,k) * Axy(i,j,k) + Gamyxy(i,j,k) * Ayy(i,j,k) + Gamzxy(i,j,k) * Ayz(i,j,k)) - p_chiy(i,j)*Axy(i,j,k)/chin1
    p_gxzy(i,j) = p_gxzy(i,j) - (  Gamxyz(i,j,k) * Axx(i,j,k) + Gamyyz(i,j,k) * Axy(i,j,k) + Gamzyz(i,j,k) * Axz(i,j,k) &
    + Gamxxy(i,j,k) * Axz(i,j,k) + Gamyxy(i,j,k) * Ayz(i,j,k) + Gamzxy(i,j,k) * Azz(i,j,k)) - p_chiy(i,j)*Axz(i,j,k)/chin1
    p_gyyy(i,j) = p_gyyy(i,j) - (  Gamxyy(i,j,k) * Axy(i,j,k) + Gamyyy(i,j,k) * Ayy(i,j,k) + Gamzyy(i,j,k) * Ayz(i,j,k) &
    + Gamxyy(i,j,k) * Axy(i,j,k) + Gamyyy(i,j,k) * Ayy(i,j,k) + Gamzyy(i,j,k) * Ayz(i,j,k)) - p_chiy(i,j)*Ayy(i,j,k)/chin1
    p_gyzy(i,j) = p_gyzy(i,j) - (  Gamxyz(i,j,k) * Axy(i,j,k) + Gamyyz(i,j,k) * Ayy(i,j,k) + Gamzyz(i,j,k) * Ayz(i,j,k) &
    + Gamxyy(i,j,k) * Axz(i,j,k) + Gamyyy(i,j,k) * Ayz(i,j,k) + Gamzyy(i,j,k) * Azz(i,j,k)) - p_chiy(i,j)*Ayz(i,j,k)/chin1
    p_gzzy(i,j) = p_gzzy(i,j) - (  Gamxyz(i,j,k) * Axz(i,j,k) + Gamyyz(i,j,k) * Ayz(i,j,k) + Gamzyz(i,j,k) * Azz(i,j,k) &
    + Gamxyz(i,j,k) * Axz(i,j,k) + Gamyyz(i,j,k) * Ayz(i,j,k) + Gamzyz(i,j,k) * Azz(i,j,k)) - p_chiy(i,j)*Azz(i,j,k)/chin1
    p_gxxz(i,j) = p_gxxz(i,j) - (  Gamxxz(i,j,k) * Axx(i,j,k) + Gamyxz(i,j,k) * Axy(i,j,k) + Gamzxz(i,j,k) * Axz(i,j,k) &
    + Gamxxz(i,j,k) * Axx(i,j,k) + Gamyxz(i,j,k) * Axy(i,j,k) + Gamzxz(i,j,k) * Axz(i,j,k)) - p_chiz(i,j)*Axx(i,j,k)/chin1
    p_gxyz(i,j) = p_gxyz(i,j) - (  Gamxyz(i,j,k) * Axx(i,j,k) + Gamyyz(i,j,k) * Axy(i,j,k) + Gamzyz(i,j,k) * Axz(i,j,k) &
    + Gamxxz(i,j,k) * Axy(i,j,k) + Gamyxz(i,j,k) * Ayy(i,j,k) + Gamzxz(i,j,k) * Ayz(i,j,k)) - p_chiz(i,j)*Axy(i,j,k)/chin1
    p_gxzz(i,j) = p_gxzz(i,j) - (  Gamxzz(i,j,k) * Axx(i,j,k) + Gamyzz(i,j,k) * Axy(i,j,k) + Gamzzz(i,j,k) * Axz(i,j,k) &
    + Gamxxz(i,j,k) * Axz(i,j,k) + Gamyxz(i,j,k) * Ayz(i,j,k) + Gamzxz(i,j,k) * Azz(i,j,k)) - p_chiz(i,j)*Axz(i,j,k)/chin1
    p_gyyz(i,j) = p_gyyz(i,j) - (  Gamxyz(i,j,k) * Axy(i,j,k) + Gamyyz(i,j,k) * Ayy(i,j,k) + Gamzyz(i,j,k) * Ayz(i,j,k) &
    + Gamxyz(i,j,k) * Axy(i,j,k) + Gamyyz(i,j,k) * Ayy(i,j,k) + Gamzyz(i,j,k) * Ayz(i,j,k)) - p_chiz(i,j)*Ayy(i,j,k)/chin1
    p_gyzz(i,j) = p_gyzz(i,j) - (  Gamxzz(i,j,k) * Axy(i,j,k) + Gamyzz(i,j,k) * Ayy(i,j,k) + Gamzzz(i,j,k) * Ayz(i,j,k) &
    + Gamxyz(i,j,k) * Axz(i,j,k) + Gamyyz(i,j,k) * Ayz(i,j,k) + Gamzyz(i,j,k) * Azz(i,j,k)) - p_chiz(i,j)*Ayz(i,j,k)/chin1
    p_gzzz(i,j) = p_gzzz(i,j) - (  Gamxzz(i,j,k) * Axz(i,j,k) + Gamyzz(i,j,k) * Ayz(i,j,k) + Gamzzz(i,j,k) * Azz(i,j,k) &
    + Gamxzz(i,j,k) * Axz(i,j,k) + Gamyzz(i,j,k) * Ayz(i,j,k) + Gamzzz(i,j,k) * Azz(i,j,k)) - p_chiz(i,j)*Azz(i,j,k)/chin1
    mx = p_gupxx(i,j)*p_gxxx(i,j) + p_gupyy(i,j)*p_gxyy(i,j) + p_gupzz(i,j)*p_gxzz(i,j) &
    +p_gupxy(i,j)*p_gxyx(i,j) + p_gupxz(i,j)*p_gxzx(i,j) + p_gupyz(i,j)*p_gxzy(i,j) &
    +p_gupxy(i,j)*p_gxxy(i,j) + p_gupxz(i,j)*p_gxxz(i,j) + p_gupyz(i,j)*p_gxyz(i,j)
    my = p_gupxx(i,j)*p_gxyx(i,j) + p_gupyy(i,j)*p_gyyy(i,j) + p_gupzz(i,j)*p_gyzz(i,j) &
    +p_gupxy(i,j)*p_gyyx(i,j) + p_gupxz(i,j)*p_gyzx(i,j) + p_gupyz(i,j)*p_gyzy(i,j) &
    +p_gupxy(i,j)*p_gxyy(i,j) + p_gupxz(i,j)*p_gxyz(i,j) + p_gupyz(i,j)*p_gyyz(i,j)
    mz = p_gupxx(i,j)*p_gxzx(i,j) + p_gupyy(i,j)*p_gyzy(i,j) + p_gupzz(i,j)*p_gzzz(i,j) &
    +p_gupxy(i,j)*p_gyzx(i,j) + p_gupxz(i,j)*p_gzzx(i,j) + p_gupyz(i,j)*p_gzzy(i,j) &
    +p_gupxy(i,j)*p_gxzy(i,j) + p_gupxz(i,j)*p_gxzz(i,j) + p_gupyz(i,j)*p_gyzz(i,j)
    movx_Res(i,j,k) = mx - F2o3*p_Kx(i,j) - F8*PI*Sx(i,j,k)
    movy_Res(i,j,k) = my - F2o3*p_Ky(i,j) - F8*PI*Sy(i,j,k)
    movz_Res(i,j,k) = mz - F2o3*p_Kz(i,j) - F8*PI*Sz(i,j,k)
    end do
    end do
    end if
  end do
  SSS(1)=SYM
  SSS(2)=SYM
  SSS(3)=SYM

  AAS(1)=ANTI
  AAS(2)=ANTI
  AAS(3)=SYM

  ASA(1)=ANTI
  ASA(2)=SYM
  ASA(3)=ANTI

  SAA(1)=SYM
  SAA(2)=ANTI
  SAA(3)=ANTI

  ASS(1)=ANTI
  ASS(2)=SYM
  ASS(3)=SYM

  SAS(1)=SYM
  SAS(2)=ANTI
  SAS(3)=SYM

  SSA(1)=SYM
  SSA(2)=SYM
  SSA(3)=ANTI

!!!!!!!!!advection term part

  call lopsided(ex,X,Y,Z,gxx,gxx_rhs,betax,betay,betaz,Symmetry,SSS)
  call lopsided(ex,X,Y,Z,gxy,gxy_rhs,betax,betay,betaz,Symmetry,AAS)
  call lopsided(ex,X,Y,Z,gxz,gxz_rhs,betax,betay,betaz,Symmetry,ASA)
  call lopsided(ex,X,Y,Z,gyy,gyy_rhs,betax,betay,betaz,Symmetry,SSS)
  call lopsided(ex,X,Y,Z,gyz,gyz_rhs,betax,betay,betaz,Symmetry,SAA)
  call lopsided(ex,X,Y,Z,gzz,gzz_rhs,betax,betay,betaz,Symmetry,SSS)

  call lopsided(ex,X,Y,Z,Axx,Axx_rhs,betax,betay,betaz,Symmetry,SSS)
  call lopsided(ex,X,Y,Z,Axy,Axy_rhs,betax,betay,betaz,Symmetry,AAS)
  call lopsided(ex,X,Y,Z,Axz,Axz_rhs,betax,betay,betaz,Symmetry,ASA)
  call lopsided(ex,X,Y,Z,Ayy,Ayy_rhs,betax,betay,betaz,Symmetry,SSS)
  call lopsided(ex,X,Y,Z,Ayz,Ayz_rhs,betax,betay,betaz,Symmetry,SAA)
  call lopsided(ex,X,Y,Z,Azz,Azz_rhs,betax,betay,betaz,Symmetry,SSS)

  call lopsided(ex,X,Y,Z,chi,chi_rhs,betax,betay,betaz,Symmetry,SSS)
  call lopsided(ex,X,Y,Z,trK,trK_rhs,betax,betay,betaz,Symmetry,SSS)

  call lopsided(ex,X,Y,Z,Gamx,Gamx_rhs,betax,betay,betaz,Symmetry,ASS)
  call lopsided(ex,X,Y,Z,Gamy,Gamy_rhs,betax,betay,betaz,Symmetry,SAS)
  call lopsided(ex,X,Y,Z,Gamz,Gamz_rhs,betax,betay,betaz,Symmetry,SSA)
!!
  call lopsided(ex,X,Y,Z,Lap,Lap_rhs,betax,betay,betaz,Symmetry,SSS)

  call lopsided(ex,X,Y,Z,betax,betax_rhs,betax,betay,betaz,Symmetry,ASS)
  call lopsided(ex,X,Y,Z,betay,betay_rhs,betax,betay,betaz,Symmetry,SAS)
  call lopsided(ex,X,Y,Z,betaz,betaz_rhs,betax,betay,betaz,Symmetry,SSA)

  call lopsided(ex,X,Y,Z,dtSfx,dtSfx_rhs,betax,betay,betaz,Symmetry,ASS)
  call lopsided(ex,X,Y,Z,dtSfy,dtSfy_rhs,betax,betay,betaz,Symmetry,SAS)
  call lopsided(ex,X,Y,Z,dtSfz,dtSfz_rhs,betax,betay,betaz,Symmetry,SSA)

  if(eps>0)then 
! usual Kreiss-Oliger dissipation      
  call kodis(ex,X,Y,Z,chi,chi_rhs,SSS,Symmetry,eps)
  call kodis(ex,X,Y,Z,trK,trK_rhs,SSS,Symmetry,eps)
  call kodis(ex,X,Y,Z,dxx,gxx_rhs,SSS,Symmetry,eps)
  call kodis(ex,X,Y,Z,gxy,gxy_rhs,AAS,Symmetry,eps)
  call kodis(ex,X,Y,Z,gxz,gxz_rhs,ASA,Symmetry,eps)
  call kodis(ex,X,Y,Z,dyy,gyy_rhs,SSS,Symmetry,eps)
  call kodis(ex,X,Y,Z,gyz,gyz_rhs,SAA,Symmetry,eps)
  call kodis(ex,X,Y,Z,dzz,gzz_rhs,SSS,Symmetry,eps)
  call kodis(ex,X,Y,Z,Axx,Axx_rhs,SSS,Symmetry,eps)
  call kodis(ex,X,Y,Z,Axy,Axy_rhs,AAS,Symmetry,eps)
  call kodis(ex,X,Y,Z,Axz,Axz_rhs,ASA,Symmetry,eps)
  call kodis(ex,X,Y,Z,Ayy,Ayy_rhs,SSS,Symmetry,eps)
  call kodis(ex,X,Y,Z,Ayz,Ayz_rhs,SAA,Symmetry,eps)
  call kodis(ex,X,Y,Z,Azz,Azz_rhs,SSS,Symmetry,eps)
  call kodis(ex,X,Y,Z,Gamx,Gamx_rhs,ASS,Symmetry,eps)
  call kodis(ex,X,Y,Z,Gamy,Gamy_rhs,SAS,Symmetry,eps)
  call kodis(ex,X,Y,Z,Gamz,Gamz_rhs,SSA,Symmetry,eps)

#if 1 
!! bam does not apply dissipation on gauge variables
  call kodis(ex,X,Y,Z,Lap,Lap_rhs,SSS,Symmetry,eps)
  call kodis(ex,X,Y,Z,betax,betax_rhs,ASS,Symmetry,eps)
  call kodis(ex,X,Y,Z,betay,betay_rhs,SAS,Symmetry,eps)
  call kodis(ex,X,Y,Z,betaz,betaz_rhs,SSA,Symmetry,eps)
  call kodis(ex,X,Y,Z,dtSfx,dtSfx_rhs,ASS,Symmetry,eps)
  call kodis(ex,X,Y,Z,dtSfy,dtSfy_rhs,SAS,Symmetry,eps)
  call kodis(ex,X,Y,Z,dtSfz,dtSfz_rhs,SSA,Symmetry,eps)
#endif

  endif


  gont = 0

  return

  contains

! ==================== Stage 1b: per-k-plane derivative helpers ====================
! fderivs_plane / fdderivs_plane replicate diff_new.f90 fderivs/fdderivs for a
! single k plane, reading the field f with a k±2 window from the 3D array.
! Boundary semantics identical to the originals:
!  - interior k in [3,ex(3)-2]: interior (i,j) 4th-order direct; i/j band via the
!    joint 4th/2nd-order condition (k part always satisfied);
!  - boundary k (1,2,ex(3)-1): joint condition per point (k4 -> 4th, else k2 -> 2nd);
!  - k=ex(3) and i=ex(1)/j=ex(2) stay 0 (never covered by the original loops);
!  - symmetry reflection when the stencil index drops below 1 (kmin/imin/jmin=-1).
  subroutine fderivs_plane(f, fx, fy, fz, s1, s2, s3)
    implicit none
    real*8, intent(in)  :: f(ex(1),ex(2),ex(3)), s1, s2, s3
    real*8, intent(out) :: fx(ex(1),ex(2)), fy(ex(1),ex(2)), fz(ex(1),ex(2))
    integer :: i, j
    fx = 0.d0
    fy = 0.d0
    fz = 0.d0
    if (k >= 3 .and. k <= ex(3)-2) then
      do j = 3, ex(2)-2
        do i = 3, ex(1)-2
          fx(i,j) = d12dx*(f(i-2,j,k) - EIT*f(i-1,j,k) + EIT*f(i+1,j,k) - f(i+2,j,k))
          fy(i,j) = d12dy*(f(i,j-2,k) - EIT*f(i,j-1,k) + EIT*f(i,j+1,k) - f(i,j+2,k))
          fz(i,j) = d12dz*(f(i,j,k-2) - EIT*f(i,j,k-1) + EIT*f(i,j,k+1) - f(i,j,k+2))
        end do
      end do
      do j = 1, ex(2)-1
        do i = 1, ex(1)-1
          if (i>=3 .and. i<=ex(1)-2 .and. j>=3 .and. j<=ex(2)-2) cycle
          if (i-2>=imin .and. i+2<=imax .and. j-2>=jmin .and. j+2<=jmax) then
            fx(i,j) = d12dx*(frx(i-2,j,k,f,s1) - EIT*frx(i-1,j,k,f,s1) + EIT*frx(i+1,j,k,f,s1) - frx(i+2,j,k,f,s1))
            fy(i,j) = d12dy*(fry(i,j-2,k,f,s2) - EIT*fry(i,j-1,k,f,s2) + EIT*fry(i,j+1,k,f,s2) - fry(i,j+2,k,f,s2))
            fz(i,j) = d12dz*(f(i,j,k-2) - EIT*f(i,j,k-1) + EIT*f(i,j,k+1) - f(i,j,k+2))
          elseif (i-1>=imin .and. i+1<=imax .and. j-1>=jmin .and. j+1<=jmax) then
            fx(i,j) = d2dx*(-frx(i-1,j,k,f,s1) + frx(i+1,j,k,f,s1))
            fy(i,j) = d2dy*(-fry(i,j-1,k,f,s2) + fry(i,j+1,k,f,s2))
            fz(i,j) = d2dz*(-f(i,j,k-1) + f(i,j,k+1))
          endif
        end do
      end do
    elseif (k < ex(3)) then
      if (k4) then
        do j = 1, ex(2)-1
          do i = 1, ex(1)-1
            if (i-2>=imin .and. i+2<=imax .and. j-2>=jmin .and. j+2<=jmax) then
              fx(i,j) = d12dx*(frx(i-2,j,k,f,s1) - EIT*frx(i-1,j,k,f,s1) + EIT*frx(i+1,j,k,f,s1) - frx(i+2,j,k,f,s1))
              fy(i,j) = d12dy*(fry(i,j-2,k,f,s2) - EIT*fry(i,j-1,k,f,s2) + EIT*fry(i,j+1,k,f,s2) - fry(i,j+2,k,f,s2))
              fz(i,j) = d12dz*(frz(i,j,k-2,f,s3) - EIT*frz(i,j,k-1,f,s3) + EIT*frz(i,j,k+1,f,s3) - frz(i,j,k+2,f,s3))
            elseif (i-1>=imin .and. i+1<=imax .and. j-1>=jmin .and. j+1<=jmax) then
              fx(i,j) = d2dx*(-frx(i-1,j,k,f,s1) + frx(i+1,j,k,f,s1))
              fy(i,j) = d2dy*(-fry(i,j-1,k,f,s2) + fry(i,j+1,k,f,s2))
              fz(i,j) = d2dz*(-frz(i,j,k-1,f,s3) + frz(i,j,k+1,f,s3))
            endif
          end do
        end do
      elseif (k2) then
        do j = 1, ex(2)-1
          do i = 1, ex(1)-1
            if (i-1>=imin .and. i+1<=imax .and. j-1>=jmin .and. j+1<=jmax) then
              fx(i,j) = d2dx*(-frx(i-1,j,k,f,s1) + frx(i+1,j,k,f,s1))
              fy(i,j) = d2dy*(-fry(i,j-1,k,f,s2) + fry(i,j+1,k,f,s2))
              fz(i,j) = d2dz*(-frz(i,j,k-1,f,s3) + frz(i,j,k+1,f,s3))
            endif
          end do
        end do
      endif
    endif
  end subroutine fderivs_plane

! 2nd derivatives (fdderivs, bam-comparison branch): joint 4th/2nd order,
! fh-padding semantics reproduced by per-axis reflection helpers.
  subroutine fdderivs_plane(f, fxx, fxy, fxz, fyy, fyz, fzz, s1, s2, s3)
    implicit none
    real*8, intent(in)  :: f(ex(1),ex(2),ex(3)), s1, s2, s3
    real*8, intent(out) :: fxx(ex(1),ex(2)), fxy(ex(1),ex(2)), fxz(ex(1),ex(2))
    real*8, intent(out) :: fyy(ex(1),ex(2)), fyz(ex(1),ex(2)), fzz(ex(1),ex(2))
    integer :: i, j
    fxx = 0.d0
    fxy = 0.d0
    fxz = 0.d0
    fyy = 0.d0
    fyz = 0.d0
    fzz = 0.d0
    if (k4) then
      do j = 1, ex(2)-1
        do i = 1, ex(1)-1
          if (i-2>=imin .and. i+2<=imax .and. j-2>=jmin .and. j+2<=jmax) then
            fxx(i,j) = Fdxdx*(-frx(i-2,j,k,f,s1) + F16*frx(i-1,j,k,f,s1) - F30*frx(i,j,k,f,s1) &
                              - frx(i+2,j,k,f,s1) + F16*frx(i+1,j,k,f,s1))
            fyy(i,j) = Fdydy*(-fry(i,j-2,k,f,s2) + F16*fry(i,j-1,k,f,s2) - F30*fry(i,j,k,f,s2) &
                              - fry(i,j+2,k,f,s2) + F16*fry(i,j+1,k,f,s2))
            fzz(i,j) = Fdzdz*(-frz(i,j,k-2,f,s3) + F16*frz(i,j,k-1,f,s3) - F30*f(i,j,k) &
                              - frz(i,j,k+2,f,s3) + F16*frz(i,j,k+1,f,s3))
            fxy(i,j) = Fdxdy*(     (frxy(i-2,j-2,k,f,s1,s2) - F8*frxy(i-1,j-2,k,f,s1,s2) + F8*frxy(i+1,j-2,k,f,s1,s2) - frxy(i+2,j-2,k,f,s1,s2)) &
                            - F8 *(frxy(i-2,j-1,k,f,s1,s2) - F8*frxy(i-1,j-1,k,f,s1,s2) + F8*frxy(i+1,j-1,k,f,s1,s2) - frxy(i+2,j-1,k,f,s1,s2)) &
                            + F8 *(frxy(i-2,j+1,k,f,s1,s2) - F8*frxy(i-1,j+1,k,f,s1,s2) + F8*frxy(i+1,j+1,k,f,s1,s2) - frxy(i+2,j+1,k,f,s1,s2)) &
                            -    (frxy(i-2,j+2,k,f,s1,s2) - F8*frxy(i-1,j+2,k,f,s1,s2) + F8*frxy(i+1,j+2,k,f,s1,s2) - frxy(i+2,j+2,k,f,s1,s2)))
            fxz(i,j) = Fdxdz*(     (frxz(i-2,j,k-2,f,s1,s3) - F8*frxz(i-1,j,k-2,f,s1,s3) + F8*frxz(i+1,j,k-2,f,s1,s3) - frxz(i+2,j,k-2,f,s1,s3)) &
                            - F8 *(frxz(i-2,j,k-1,f,s1,s3) - F8*frxz(i-1,j,k-1,f,s1,s3) + F8*frxz(i+1,j,k-1,f,s1,s3) - frxz(i+2,j,k-1,f,s1,s3)) &
                            + F8 *(frxz(i-2,j,k+1,f,s1,s3) - F8*frxz(i-1,j,k+1,f,s1,s3) + F8*frxz(i+1,j,k+1,f,s1,s3) - frxz(i+2,j,k+1,f,s1,s3)) &
                            -    (frxz(i-2,j,k+2,f,s1,s3) - F8*frxz(i-1,j,k+2,f,s1,s3) + F8*frxz(i+1,j,k+2,f,s1,s3) - frxz(i+2,j,k+2,f,s1,s3)))
            fyz(i,j) = Fdydz*(     (fryz(i,j-2,k-2,f,s2,s3) - F8*fryz(i,j-1,k-2,f,s2,s3) + F8*fryz(i,j+1,k-2,f,s2,s3) - fryz(i,j+2,k-2,f,s2,s3)) &
                            - F8 *(fryz(i,j-2,k-1,f,s2,s3) - F8*fryz(i,j-1,k-1,f,s2,s3) + F8*fryz(i,j+1,k-1,f,s2,s3) - fryz(i,j+2,k-1,f,s2,s3)) &
                            + F8 *(fryz(i,j-2,k+1,f,s2,s3) - F8*fryz(i,j-1,k+1,f,s2,s3) + F8*fryz(i,j+1,k+1,f,s2,s3) - fryz(i,j+2,k+1,f,s2,s3)) &
                            -    (fryz(i,j-2,k+2,f,s2,s3) - F8*fryz(i,j-1,k+2,f,s2,s3) + F8*fryz(i,j+1,k+2,f,s2,s3) - fryz(i,j+2,k+2,f,s2,s3)))
          elseif (i-1>=imin .and. i+1<=imax .and. j-1>=jmin .and. j+1<=jmax) then
            fxx(i,j) = Sdxdx*(frx(i-1,j,k,f,s1) - TWO*frx(i,j,k,f,s1) + frx(i+1,j,k,f,s1))
            fyy(i,j) = Sdydy*(fry(i,j-1,k,f,s2) - TWO*fry(i,j,k,f,s2) + fry(i,j+1,k,f,s2))
            fzz(i,j) = Sdzdz*(frz(i,j,k-1,f,s3) - TWO*f(i,j,k) + frz(i,j,k+1,f,s3))
            fxy(i,j) = Sdxdy*(frxy(i-1,j-1,k,f,s1,s2) - frxy(i+1,j-1,k,f,s1,s2) - frxy(i-1,j+1,k,f,s1,s2) + frxy(i+1,j+1,k,f,s1,s2))
            fxz(i,j) = Sdxdz*(frxz(i-1,j,k-1,f,s1,s3) - frxz(i+1,j,k-1,f,s1,s3) - frxz(i-1,j,k+1,f,s1,s3) + frxz(i+1,j,k+1,f,s1,s3))
            fyz(i,j) = Sdydz*(fryz(i,j-1,k-1,f,s2,s3) - fryz(i,j+1,k-1,f,s2,s3) - fryz(i,j-1,k+1,f,s2,s3) + fryz(i,j+1,k+1,f,s2,s3))
          endif
        end do
      end do
    elseif (k2) then
      do j = 1, ex(2)-1
        do i = 1, ex(1)-1
          if (i-1>=imin .and. i+1<=imax .and. j-1>=jmin .and. j+1<=jmax) then
            fxx(i,j) = Sdxdx*(frx(i-1,j,k,f,s1) - TWO*frx(i,j,k,f,s1) + frx(i+1,j,k,f,s1))
            fyy(i,j) = Sdydy*(fry(i,j-1,k,f,s2) - TWO*fry(i,j,k,f,s2) + fry(i,j+1,k,f,s2))
            fzz(i,j) = Sdzdz*(frz(i,j,k-1,f,s3) - TWO*f(i,j,k) + frz(i,j,k+1,f,s3))
            fxy(i,j) = Sdxdy*(frxy(i-1,j-1,k,f,s1,s2) - frxy(i+1,j-1,k,f,s1,s2) - frxy(i-1,j+1,k,f,s1,s2) + frxy(i+1,j+1,k,f,s1,s2))
            fxz(i,j) = Sdxdz*(frxz(i-1,j,k-1,f,s1,s3) - frxz(i+1,j,k-1,f,s1,s3) - frxz(i-1,j,k+1,f,s1,s3) + frxz(i+1,j,k+1,f,s1,s3))
            fyz(i,j) = Sdydz*(fryz(i,j-1,k-1,f,s2,s3) - fryz(i,j+1,k-1,f,s2,s3) - fryz(i,j-1,k+1,f,s2,s3) + fryz(i,j+1,k+1,f,s2,s3))
          endif
        end do
      end do
    endif
  end subroutine fdderivs_plane

! ---- reflection helpers (fh / frx/fry/frz padding semantics) ----
  function frx(a, jj, kk, ff, s1) result(r)
    integer, intent(in) :: a, jj, kk
    real*8, intent(in)  :: ff(ex(1),ex(2),ex(3)), s1
    real*8 :: r
    if (a >= 1) then
      r = ff(a, jj, kk)
    else
      r = ff(-a+1, jj, kk)*s1
    endif
  end function frx

  function fry(ii, b, kk, ff, s2) result(r)
    integer, intent(in) :: ii, b, kk
    real*8, intent(in)  :: ff(ex(1),ex(2),ex(3)), s2
    real*8 :: r
    if (b >= 1) then
      r = ff(ii, b, kk)
    else
      r = ff(ii, -b+1, kk)*s2
    endif
  end function fry

  function frz(ii, jj, c, ff, s3) result(r)
    integer, intent(in) :: ii, jj, c
    real*8, intent(in)  :: ff(ex(1),ex(2),ex(3)), s3
    real*8 :: r
    if (c >= 1) then
      r = ff(ii, jj, c)
    else
      r = ff(ii, jj, -c+1)*s3
    endif
  end function frz

  function frxy(a, b, kk, ff, s1, s2) result(r)
    integer, intent(in) :: a, b, kk
    real*8, intent(in)  :: ff(ex(1),ex(2),ex(3)), s1, s2
    real*8 :: r
    integer :: a2, b2
    a2 = a
    if (a < 1) a2 = -a+1
    b2 = b
    if (b < 1) b2 = -b+1
    r = ff(a2, b2, kk)
    if (a < 1) r = r*s1
    if (b < 1) r = r*s2
  end function frxy

  function frxz(a, jj, c, ff, s1, s3) result(r)
    integer, intent(in) :: a, jj, c
    real*8, intent(in)  :: ff(ex(1),ex(2),ex(3)), s1, s3
    real*8 :: r
    integer :: a2, c2
    a2 = a
    if (a < 1) a2 = -a+1
    c2 = c
    if (c < 1) c2 = -c+1
    r = ff(a2, jj, c2)
    if (a < 1) r = r*s1
    if (c < 1) r = r*s3
  end function frxz

  function fryz(ii, b, c, ff, s2, s3) result(r)
    integer, intent(in) :: ii, b, c
    real*8, intent(in)  :: ff(ex(1),ex(2),ex(3)), s2, s3
    real*8 :: r
    integer :: b2, c2
    b2 = b
    if (b < 1) b2 = -b+1
    c2 = c
    if (c < 1) c2 = -c+1
    r = ff(ii, b2, c2)
    if (b < 1) r = r*s2
    if (c < 1) r = r*s3
  end function fryz


  end function compute_rhs_bssn