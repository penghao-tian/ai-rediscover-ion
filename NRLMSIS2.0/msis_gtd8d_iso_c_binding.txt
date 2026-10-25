module msis_module
  use iso_c_binding, only: c_int, c_float
  use msis_constants, only: rp, dmissing
  use msis_init, only: msisinit
  use msis_calc, only: msiscalc
  implicit none
contains
  subroutine gtd8d(iyd, sec, alt, glat, glong, stl, f107a, f107, ap, mass, d, t) bind(C, name="gtd8d")
    implicit none
    ! MSIS Legacy subroutine arguments
    integer(c_int), intent(in)         :: iyd
    real(c_float), intent(in)          :: sec
    real(c_float), intent(in)          :: alt
    real(c_float), intent(in)          :: glat
    real(c_float), intent(in)          :: glong
    real(c_float), intent(in)          :: stl
    real(c_float), intent(in)          :: f107a
    real(c_float), intent(in)          :: f107
    real(c_float), intent(in)          :: ap(7)
    integer(c_int), intent(in)         :: mass
    real(c_float), intent(out)         :: d(9), t(2)

    ! MSIS 1.97 subroutine arguments
    real(kind=rp)               :: xday
    real(kind=rp)               :: xutsec
    real(kind=rp)               :: xalt
    real(kind=rp)               :: xlat
    real(kind=rp)               :: xlon
    real(kind=rp)               :: xsfluxavg, xsflux
    real(kind=rp)               :: xap(1:7)
    real(kind=rp)               :: xtn
    real(kind=rp)               :: xdn(1:10)
    real(kind=rp)               :: xtex

    ! Convert the legacy input arguments to the new interface values and precision
    xday = mod(iyd,1000)
    xutsec = sec
    xalt = alt
    xlat = glat
    xlon = glong
    xsfluxavg = f107a
    xsflux = f107
    xap = ap

    ! Call the new subroutine
    call msiscalc(xday, xutsec, xalt, xlat, xlon, xsfluxavg, xsflux, xap, xtn, xdn, tex=xtex)

    ! Convert the output arguments to the legacy format (mks to cgs, re-order species)
    t(1) = sngl(xtex)    ! Exospheric temperature
    t(2) = sngl(xtn)     ! Temperature at altitude
    where (xdn .ne. dmissing) xdn = xdn*1d-6
    if (xdn(1) .ne. dmissing) xdn(1) = xdn(1)*1e3_rp
    d(1) = sngl(xdn(5))  ! [He]
    d(2) = sngl(xdn(4))  ! [O]
    d(3) = sngl(xdn(2))  ! [N2]
    d(4) = sngl(xdn(3))  ! [O2]
    d(5) = sngl(xdn(7))  ! [Ar]
    d(6) = sngl(xdn(1))  ! Mass density
    d(7) = sngl(xdn(6))  ! [H]
    d(8) = sngl(xdn(8))  ! [N]
    d(9) = sngl(xdn(9))  ! [Anomalous O]

    return
  end subroutine gtd8d
end module msis_module
