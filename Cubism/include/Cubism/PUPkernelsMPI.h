#pragma once

namespace cubism {

template <typename Real>
inline void pack(const Real *const srcbase, Real *const dst, const unsigned int gptfloats, int *selected_components, const int ncomponents, const int xstart, const int ystart, const int zstart, const int xend, const int yend, const int zend, const int BSX, const int BSY)
{
   if (gptfloats == 1)
   {
      const int mod = (xend-xstart)%4;
      for (int idst = 0, iz = zstart; iz < zend; ++iz)
      for (int iy = ystart; iy < yend; ++iy)
      {
         for (int ix = xstart; ix < xend-mod; ix+=4, idst+=4)
         {
            dst[idst+0] = srcbase[ix+0 + BSX * (iy + BSY * iz)];
            dst[idst+1] = srcbase[ix+1 + BSX * (iy + BSY * iz)];
            dst[idst+2] = srcbase[ix+2 + BSX * (iy + BSY * iz)];
            dst[idst+3] = srcbase[ix+3 + BSX * (iy + BSY * iz)];
         }
         for (int ix = xend-mod; ix < xend; ix++, idst++)
         {
            dst[idst] = srcbase[ix + BSX * (iy + BSY * iz)];
         }
      }
   }
   else
   {
      for (int idst = 0, iz = zstart; iz < zend; ++iz)
      for (int iy = ystart; iy < yend; ++iy)
      for (int ix = xstart; ix < xend; ++ix)
      {
         const Real *src = srcbase + gptfloats * (ix + BSX * (iy + BSY * iz));
         for (int ic = 0; ic < ncomponents; ic++, idst++) dst[idst] = src[selected_components[ic]];
      }
   }
}

template <typename Real>
inline void unpack_subregion(const Real *const pack, Real *const dstbase, const unsigned int gptfloats, const int *const selected_components, const int ncomponents, const int srcxstart, const int srcystart, const int srczstart, const int LX, const int LY, const int dstxstart, const int dstystart, const int dstzstart, const int dstxend, const int dstyend, const int dstzend, const int xsize, const int ysize, const int zsize)
{
   if (gptfloats == 1)
   {
      const int mod = (dstxend-dstxstart)%4;
      for (int zd = dstzstart; zd < dstzend; ++zd)
      for (int yd = dstystart; yd < dstyend; ++yd)
      {
         const int offset     = - dstxstart + srcxstart + LX * (yd - dstystart + srcystart + LY * (zd - dstzstart + srczstart));
         const int offset_dst = xsize * (yd + ysize * zd);
         for (int xd = dstxstart; xd < dstxend-mod; xd+=4)
         {
            dstbase[xd + 0 + offset_dst] = pack[xd + 0 + offset];
            dstbase[xd + 1 + offset_dst] = pack[xd + 1 + offset];
            dstbase[xd + 2 + offset_dst] = pack[xd + 2 + offset];
            dstbase[xd + 3 + offset_dst] = pack[xd + 3 + offset];
         }
         for (int xd = dstxend-mod; xd < dstxend; ++xd)
         {
            dstbase[xd + offset_dst] = pack[xd +offset];
         }
      }
   }
   else
   {
      for (int zd = dstzstart; zd < dstzend; ++zd)
      for (int yd = dstystart; yd < dstyend; ++yd)
      for (int xd = dstxstart; xd < dstxend; ++xd)
      {
         Real *const dst = dstbase + gptfloats * (xd + xsize * (yd + ysize * zd));
         const Real *src = pack + ncomponents * (xd - dstxstart + srcxstart + LX * (yd - dstystart + srcystart + LY * (zd - dstzstart + srczstart)));
         for (int c = 0; c < ncomponents; ++c) dst[selected_components[c]] = src[c];
      }
   }
}

}//namespace cubism