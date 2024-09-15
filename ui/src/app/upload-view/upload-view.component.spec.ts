import { ComponentFixture, TestBed } from '@angular/core/testing';

import { UploadViewComponent } from './upload-view.component';

describe('UploadViewComponent', () => {
  let component: UploadViewComponent;
  let fixture: ComponentFixture<UploadViewComponent>;

  beforeEach(async () => {
    await TestBed.configureTestingModule({
      declarations: [UploadViewComponent]
    })
    .compileComponents();
    
    fixture = TestBed.createComponent(UploadViewComponent);
    component = fixture.componentInstance;
    fixture.detectChanges();
  });

  it('should create', () => {
    expect(component).toBeTruthy();
  });
});
